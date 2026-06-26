import argparse
import shutil
import warnings
from pathlib import Path

import cooler
import matplotlib

matplotlib.use("Agg")
import numpy as np
import scipy.sparse as sps

from hiarch_defaults import (
    DATA_DIR,
    DEFAULT_CHECKERBOARD_MIN_CHRO_BINS,
    DEFAULT_NORMDIS_MIN_CHRO_BINS,
    INPUT_DIR,
    assess_resolution_filtering,
    balance_figure_dir,
    balanced_mcool_path,
    choose_filter_passing_resolution,
    iter_mcool_files,
    resolution_file_path,
    resolve_resolution,
    species_dir,
    species_from_mcool,
)


def balance_min_nnz_attempts(initial_min_nnz):
    attempts = [initial_min_nnz]
    for value in (10, 5, 2, 1, 0):
        if value < initial_min_nnz and value not in attempts:
            attempts.append(value)
    return attempts


def clean_balance_stats(stats):
    clean_stats = {}
    for key, value in stats.items():
        if hasattr(value, "item"):
            value = value.item()
        clean_stats[key] = value
    return clean_stats


def add_publish_to_path():
    import sys

    publish_dir = Path(__file__).resolve().parents[1] / "publish"
    publish_dir_str = str(publish_dir)
    if publish_dir_str not in sys.path:
        sys.path.insert(0, publish_dir_str)


def get_balanced_pixels(clr, weights=None):
    pixels = clr.pixels()[:]
    rows = pixels["bin1_id"].to_numpy()
    cols = pixels["bin2_id"].to_numpy()
    counts = pixels["count"].to_numpy()
    if weights is None:
        weights = clr.bins()["weight"][:].to_numpy()

    values = counts.astype(float) * weights[rows] * weights[cols]
    keep = np.isfinite(values) & (values > 0)
    return rows[keep], cols[keep], values[keep]


def count_balanced_pixels(clr, weights=None):
    _, _, values = get_balanced_pixels(clr, weights=weights)
    return int(values.size)


def get_balanced_matrix(clr, weights=None):
    rows, cols, values = get_balanced_pixels(clr, weights=weights)
    return sps.coo_matrix(
        (values, (rows, cols)),
        shape=(clr.info["nbins"], clr.info["nbins"]),
    )


def get_raw_matrix(clr):
    pixels = clr.pixels()[:]
    rows = pixels["bin1_id"].to_numpy()
    cols = pixels["bin2_id"].to_numpy()
    counts = pixels["count"].to_numpy()
    return sps.coo_matrix(
        (counts, (rows, cols)),
        shape=(clr.info["nbins"], clr.info["nbins"]),
    )


def store_ice_weights(clr, weights, stats, store_name="weight"):
    with clr.open("r+") as group:
        if store_name in group["bins"]:
            del group["bins"][store_name]
        dataset = group["bins"].create_dataset(
            store_name,
            data=weights,
            compression="gzip",
            compression_opts=6,
        )
        dataset.attrs.update(stats)


def balance_cooler_with_fallbacks(clr, initial_min_nnz):
    last_stats = None
    last_finite_weights = 0
    last_balanced_pixels = 0

    attempts = balance_min_nnz_attempts(initial_min_nnz)
    for attempt_idx, min_nnz in enumerate(attempts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weights, stats = cooler.balance_cooler(
                clr,
                ignore_diags=2,
                mad_max=5,
                min_nnz=min_nnz,
                min_count=0,
                tol=1e-5,
                max_iters=200,
                chunksize=10_000_000,
                store=False,
            )
        stats = clean_balance_stats(stats)
        finite_weights = int(np.isfinite(weights).sum())
        balanced_pixels = count_balanced_pixels(clr, weights=weights)
        last_stats = stats
        last_finite_weights = finite_weights
        last_balanced_pixels = balanced_pixels

        if finite_weights > 0 and balanced_pixels > 0:
            store_ice_weights(clr, weights, stats)
            return weights, stats, finite_weights, balanced_pixels

        next_action = "retrying." if attempt_idx < len(attempts) - 1 else "no fallback remains."
        print(
            f"ICE balancing with min_nnz={min_nnz} produced no usable balanced contacts "
            f"(finite weights: {finite_weights}/{len(weights)}, contacts: {balanced_pixels}); {next_action}"
        )

    raise RuntimeError(
        "ICE balancing produced no usable finite positive contacts. "
        f"Last finite weights: {last_finite_weights}/{clr.info['nbins']}; "
        f"last balanced contacts: {last_balanced_pixels}; last stats: {last_stats}"
    )


def plot_balance_maps(output_mcool, resolution, figure_dir):
    add_publish_to_path()
    from new_hic_class import GenomeIndex, SpsHiCMtx
    from iutils.plot_heatmap import heatmap_plot

    figure_dir.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(f"{output_mcool}::resolutions/{resolution}")
    bins = clr.bins()[:]
    window = bins[["chrom", "start", "end"]].copy()
    window.rename(columns={"chrom": "chr"}, inplace=True)
    window.insert(0, "index", np.arange(window.shape[0]))
    window = window[["chr", "start", "end", "index"]]
    gen_index = GenomeIndex(window)

    raw_mtx = SpsHiCMtx(get_raw_matrix(clr), row_window=gen_index, mtx_type="triu", has_neg=False)
    heatmap_plot(raw_mtx, cmap="vlag", out_file=str(figure_dir / "raw_count_map.png"))

    balanced_mtx = SpsHiCMtx(get_balanced_matrix(clr), row_window=gen_index, mtx_type="triu", has_neg=False)
    heatmap_plot(balanced_mtx, cmap="vlag", out_file=str(figure_dir / "ice_balanced_map.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcool", default=None)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(INPUT_DIR))
    parser.add_argument("--species", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--resolution",
        default="auto",
        help="Resolution/bin size to balance. Use 'auto' to choose the coarsest bin size in the mcool.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument(
        "--balance-min-nnz",
        type=int,
        default=10,
        help=(
            "Initial cooler min_nnz filter for ICE balancing. If this produces no usable "
            "weights, lower values are retried automatically."
        ),
    )
    parser.add_argument(
        "--normdis-min-chro-bins",
        type=int,
        default=DEFAULT_NORMDIS_MIN_CHRO_BINS,
        help="Auto-select resolutions that keep at least one chromosome with this many bins after NormDis filtering.",
    )
    parser.add_argument(
        "--checkerboard-min-chro-bins",
        type=int,
        default=DEFAULT_CHECKERBOARD_MIN_CHRO_BINS,
        help="Prefer resolutions that also keep at least one chromosome with this many bins for Checkerboard.",
    )
    args = parser.parse_args()

    input_files = [Path(args.mcool)] if args.mcool is not None else iter_mcool_files(args.data_dir)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No .mcool files found in {args.data_dir}")
    if args.output is not None and len(input_files) > 1:
        raise ValueError("--output can only be used when balancing a single --mcool file")

    for input_mcool in input_files:
        species = args.species or species_from_mcool(input_mcool)
        output_mcool = Path(args.output) if args.output is not None else balanced_mcool_path(species, args.output_dir)
        resolution_file = resolution_file_path(species, args.output_dir)
        figure_dir = Path(args.figure_dir) if args.figure_dir is not None else balance_figure_dir(species, args.output_dir)

        output_mcool.parent.mkdir(parents=True, exist_ok=True)

        if input_mcool.resolve() != output_mcool.resolve() and (args.force or not output_mcool.exists()):
            shutil.copy2(input_mcool, output_mcool)

        try:
            if args.resolution == "auto":
                resolution, assessments = choose_filter_passing_resolution(
                    output_mcool,
                    normdis_min_chro_bins=args.normdis_min_chro_bins,
                    checkerboard_min_chro_bins=args.checkerboard_min_chro_bins,
                )
            else:
                resolution = resolve_resolution(
                    output_mcool,
                    args.resolution,
                    normdis_min_chro_bins=args.normdis_min_chro_bins,
                    checkerboard_min_chro_bins=args.checkerboard_min_chro_bins,
                )
                assessments = [
                    assess_resolution_filtering(
                        output_mcool,
                        resolution,
                        normdis_min_chro_bins=args.normdis_min_chro_bins,
                        checkerboard_min_chro_bins=args.checkerboard_min_chro_bins,
                    )
                ]
        except ValueError as e:
            if "No resolution passes NormDis filtering" in str(e):
                print(f"Warning: Skipping sample {input_mcool.name}, data too sparse for balancing")
                print(f"Reason: {e}")
                continue  # Skip current sample, continue to next one
            else:
                raise  # Re-raise other errors

        qc_file = species_dir(species, args.output_dir) / "resolution_qc.tsv"
        qc_file.parent.mkdir(parents=True, exist_ok=True)
        with qc_file.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("resolution\traw_bins\traw_chroms\tkept_chroms_after_zero_name\tmax_bins_after_low_coverage\tnormdis_passing_chroms\tcheckerboard_passing_chroms\tpasses_normdis\tpasses_checkerboard\n")
            for item in assessments:
                handle.write(
                    f"{item['resolution']}\t{item['raw_bins']}\t{item['raw_chroms']}\t"
                    f"{item['kept_chroms_after_zero_name']}\t{item['max_bins_after_low_coverage']}\t"
                    f"{item['normdis_passing_chroms']}\t{item['checkerboard_passing_chroms']}\t"
                    f"{item['passes_normdis']}\t{item['passes_checkerboard']}\n"
                )

        resolution_file.parent.mkdir(parents=True, exist_ok=True)
        resolution_file.write_text(f"{resolution}\n", encoding="ascii")
        clr = cooler.Cooler(f"{output_mcool}::resolutions/{resolution}")
        if "weight" in set(clr.bins().columns) and not args.force:
            weights = clr.bins()["weight"][:].to_numpy()
            finite_weights = int(np.isfinite(weights).sum())
            balanced_pixels = count_balanced_pixels(clr, weights=weights)
            if finite_weights > 0 and balanced_pixels > 0:
                print(f"Using existing ICE weights: {output_mcool}::resolutions/{resolution}")
                if not args.no_plot:
                    plot_balance_maps(output_mcool, resolution, figure_dir)
                    print(f"Balance figures: {figure_dir}")
                continue
            print(
                f"Existing ICE weights are unusable for {output_mcool}::resolutions/{resolution} "
                f"(finite weights: {finite_weights}/{len(weights)}, contacts: {balanced_pixels}); recomputing."
            )

        weights, stats, finite_weights, balanced_pixels = balance_cooler_with_fallbacks(
            clr,
            args.balance_min_nnz,
        )

        print(f"Species: {species}")
        print(f"Balanced resolution: {resolution}")
        print(f"Finite weights: {finite_weights}/{len(weights)}")
        print(f"Balanced contacts: {balanced_pixels}")
        if not stats.get("converged", False):
            print(f"Warning: ICE balancing did not converge. Stats: {stats}")
        print(f"Output mcool: {output_mcool}")

        if not args.no_plot:
            plot_balance_maps(output_mcool, resolution, figure_dir)
            print(f"Balance figures: {figure_dir}")


if __name__ == "__main__":
    main()
