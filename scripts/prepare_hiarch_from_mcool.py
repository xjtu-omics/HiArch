import argparse
import shutil
from pathlib import Path

import cooler
import h5py
import numpy as np

from hiarch_defaults import (
    DATA_DIR,
    INPUT_DIR,
    balanced_mcool_path,
    iter_mcool_files,
    resolution_file_path,
    resolve_resolution,
    species_from_mcool,
)


def choose_mcool(input_mcool, balanced_mcool):
    input_mcool = Path(input_mcool)
    if balanced_mcool is None:
        return input_mcool

    balanced_mcool = Path(balanced_mcool)
    balanced_mcool.parent.mkdir(parents=True, exist_ok=True)
    if input_mcool.resolve() != balanced_mcool.resolve():
        shutil.copy2(input_mcool, balanced_mcool)
    return balanced_mcool


def ensure_ice_weights(mcool_path, resolution, force=False):
    uri = f"{mcool_path}::resolutions/{resolution}"
    clr = cooler.Cooler(uri)
    bins_columns = set(clr.bins().columns)
    if "weight" in bins_columns and not force:
        return uri

    weights, stats = cooler.balance_cooler(
        clr,
        ignore_diags=2,
        mad_max=5,
        min_nnz=10,
        min_count=0,
        tol=1e-5,
        max_iters=200,
        chunksize=10_000_000,
        store=True,
        store_name="weight",
    )
    finite_weights = int(np.isfinite(weights).sum())
    if finite_weights == 0:
        raise RuntimeError(f"ICE balancing produced no finite weights. Stats: {stats}")
    return uri


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcool", default=None)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--resolution",
        default=None,
        help="Resolution/bin size to export. Use 'auto' to choose the largest value in the mcool.",
    )
    parser.add_argument("--outdir", default=str(INPUT_DIR))
    parser.add_argument("--species", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing generated .mtx/.bed files in each species sps_mtx directory.",
    )
    parser.add_argument(
        "--ice-balance",
        action="store_true",
        help="Run ICE balancing with cooler and export balanced values.",
    )
    parser.add_argument(
        "--balanced-values",
        action="store_true",
        default=True,
        help="Export count * weight[bin1] * weight[bin2] from an already balanced mcool.",
    )
    parser.add_argument(
        "--raw-counts",
        action="store_false",
        dest="balanced_values",
        help="Export raw pixel counts instead of ICE-balanced values.",
    )
    parser.add_argument(
        "--force-balance",
        action="store_true",
        help="Recompute ICE weights even if bins/weight already exists.",
    )
    parser.add_argument(
        "--balanced-mcool",
        default=None,
        help="Optional copy of the mcool to modify with ICE weights, preserving the original.",
    )
    args = parser.parse_args()

    input_files = []
    if args.mcool is not None:
        input_files = [Path(args.mcool)]
    else:
        for raw_mcool in iter_mcool_files(args.data_dir):
            species = species_from_mcool(raw_mcool)
            input_files.append(balanced_mcool_path(species, args.outdir))
    if len(input_files) == 0:
        raise FileNotFoundError(f"No .mcool files found in {args.data_dir}")

    for input_mcool in input_files:
        species = args.species or species_from_mcool(input_mcool)
        requested_resolution = args.resolution
        resolution_file = resolution_file_path(species, args.outdir)
        if requested_resolution is None:
            requested_resolution = resolution_file.read_text(encoding="ascii").strip() if resolution_file.exists() else "auto"

        outdir = Path(args.outdir)
        species_dir = outdir / species
        sps_dir = species_dir / "sps_mtx"
        sps_dir.mkdir(parents=True, exist_ok=True)
        if not args.keep_existing:
            for old_file in list(sps_dir.glob("*_normalized.mtx")) + list(sps_dir.glob("*.window.bed")):
                old_file.unlink()

        mcool_path = choose_mcool(input_mcool, args.balanced_mcool)
        resolution = resolve_resolution(mcool_path, requested_resolution)
        sample = args.sample or f"{species}_{resolution}"
        use_balanced_values = args.ice_balance or args.balanced_values
        if args.ice_balance:
            ensure_ice_weights(mcool_path, resolution, force=args.force_balance)

        with h5py.File(mcool_path, "r") as mcool:
            group = mcool[f"resolutions/{resolution}"]
            chrom_names = [decode_name(value) for value in group["chroms"]["name"][:]]
            chrom_lengths = [int(v) for v in group["chroms"]["length"][:]]
            chrom_sizes_file = species_dir / f"{species}.chrom.sizes"
            with chrom_sizes_file.open("w", encoding="ascii", newline="\n") as handle:
                for name, length in zip(chrom_names, chrom_lengths):
                    handle.write(f"{name}\t{length}\n")

            bin_chrom = group["bins"]["chrom"][:]
            bin_start = group["bins"]["start"][:]
            bin_end = group["bins"]["end"][:]

            window_file = sps_dir / f"{sample}.window.bed"
            with window_file.open("w", encoding="ascii", newline="\n") as handle:
                for idx, (chrom_idx, start, end) in enumerate(zip(bin_chrom, bin_start, bin_end)):
                    handle.write(f"{chrom_names[int(chrom_idx)]}\t{int(start)}\t{int(end)}\t{idx}\n")

            mtx_file = sps_dir / f"{sample}_normalized.mtx"
            rows = group["pixels"]["bin1_id"][:]
            cols = group["pixels"]["bin2_id"][:]
            counts = group["pixels"]["count"][:]
            if use_balanced_values:
                if "weight" not in group["bins"]:
                    raise RuntimeError(f"Balanced export requested for {species}, but bins/weight was not found.")
                weights = group["bins"]["weight"][:]
                values = counts.astype(float) * weights[rows] * weights[cols]
                keep = np.isfinite(values) & (values > 0)
            else:
                values = counts
                keep = np.ones(len(counts), dtype=bool)

            with mtx_file.open("w", encoding="ascii", newline="\n") as handle:
                for row, col, value in zip(rows[keep], cols[keep], values[keep]):
                    handle.write(f"{int(row)}\t{int(col)}\t{float(value):.10g}\n")

        readme = species_dir / "README_input.txt"
        value_source = "ICE-balanced values: count * weight[bin1] * weight[bin2]." if use_balanced_values else "Raw pixel counts."
        readme.write_text(
            "\n".join(
                [
                    f"HiArch input generated from {mcool_path}.",
                    f"Resolution: {resolution} bp.",
                    value_source,
                    "Chromosome names and lengths were read directly from the mcool chroms table.",
                    "",
                ]
            ),
            encoding="ascii",
        )
        print(f"Prepared HiArch input: {species} ({resolution})")


if __name__ == "__main__":
    main()
