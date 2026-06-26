from pathlib import Path
import re

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
INPUT_DIR = ROOT_DIR / "Output"
PARAMETERS_FILE = INPUT_DIR / "parameters.txt"
PYTHON = "python"
SCRIPT_DIR = Path("publish") / "HiArch"
DEFAULT_NORMDIS_MIN_CHRO_BINS = 21
DEFAULT_CHECKERBOARD_MIN_CHRO_BINS = 50
LOW_COVERAGE_PERCENTILE = 5
MIN_CHRO_NONZERO_RATIO = 0.01


def species_from_mcool(mcool_path):
    name = Path(mcool_path).name
    if name.endswith(".ice.mcool"):
        return name[: -len(".ice.mcool")]
    if name.endswith(".mcool"):
        return name[: -len(".mcool")]
    return Path(mcool_path).stem


def iter_mcool_files(data_dir=DATA_DIR):
    return sorted(Path(data_dir).glob("*.mcool"))


def species_dir(species, output_dir=INPUT_DIR):
    return Path(output_dir) / species


def balanced_mcool_path(species, output_dir=INPUT_DIR):
    return species_dir(species, output_dir) / f"{species}.ice.mcool"


def resolution_file_path(species, output_dir=INPUT_DIR):
    return species_dir(species, output_dir) / "resolution.txt"


def balance_figure_dir(species, output_dir=INPUT_DIR):
    return species_dir(species, output_dir) / "balance_figure"


def _decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _filter_chroms_by_name(chrom_names, chrom_bin_counts, kept_chroms):
    if len(kept_chroms) == 0:
        return kept_chroms
    max_len_chrom = max(kept_chroms, key=lambda chrom: chrom_bin_counts[chrom])
    match = re.search(r"([A-Z]*)\d", max_len_chrom)
    if match is None:
        return kept_chroms
    prefix = match.group(1)
    return [chrom for chrom in kept_chroms if chrom.startswith(prefix)]


def assess_resolution_filtering(
    mcool_path,
    resolution,
    normdis_min_chro_bins=DEFAULT_NORMDIS_MIN_CHRO_BINS,
    checkerboard_min_chro_bins=DEFAULT_CHECKERBOARD_MIN_CHRO_BINS,
):
    with h5py.File(mcool_path, "r") as mcool:
        group = mcool[f"resolutions/{resolution}"]
        chrom_names = [_decode_name(value) for value in group["chroms"]["name"][:]]
        bin_chrom = group["bins"]["chrom"][:].astype(int)
        rows = group["pixels"]["bin1_id"][:].astype(int)
        cols = group["pixels"]["bin2_id"][:].astype(int)
        counts = group["pixels"]["count"][:].astype(float)

    n_bins = len(bin_chrom)
    chrom_bin_counts = {
        chrom: int(np.count_nonzero(bin_chrom == chrom_idx))
        for chrom_idx, chrom in enumerate(chrom_names)
    }

    coverage = np.zeros(n_bins, dtype=float)
    np.add.at(coverage, rows, counts)
    np.add.at(coverage, cols, counts)

    kept_chroms = []
    for chrom_idx, chrom in enumerate(chrom_names):
        chrom_bins = np.nonzero(bin_chrom == chrom_idx)[0]
        chrom_len = len(chrom_bins)
        if chrom_len == 0:
            continue
        intra = (bin_chrom[rows] == chrom_idx) & (bin_chrom[cols] == chrom_idx)
        total_triu = int((chrom_len * chrom_len - chrom_len) / 2 + chrom_len)
        nonzero_ratio = np.count_nonzero(intra) / total_triu if total_triu > 0 else 0
        if nonzero_ratio >= MIN_CHRO_NONZERO_RATIO:
            kept_chroms.append(chrom)

    kept_chroms = _filter_chroms_by_name(chrom_names, chrom_bin_counts, kept_chroms)
    kept_chrom_idxs = {chrom_names.index(chrom) for chrom in kept_chroms}
    kept_bin_mask = np.array([chrom_idx in kept_chrom_idxs for chrom_idx in bin_chrom])
    kept_bins = np.nonzero(kept_bin_mask)[0]

    post_low_coverage_counts = {chrom: 0 for chrom in kept_chroms}
    if len(kept_bins) > 0:
        coverage_threshold = np.percentile(coverage[kept_bins], LOW_COVERAGE_PERCENTILE)
        high_coverage_mask = kept_bin_mask & (coverage > coverage_threshold)
        for chrom_idx, chrom in enumerate(chrom_names):
            if chrom in post_low_coverage_counts:
                post_low_coverage_counts[chrom] = int(np.count_nonzero(high_coverage_mask & (bin_chrom == chrom_idx)))

    normdis_chroms = [
        chrom for chrom, bin_count in post_low_coverage_counts.items()
        if bin_count >= normdis_min_chro_bins
    ]
    checkerboard_chroms = [
        chrom for chrom, bin_count in post_low_coverage_counts.items()
        if bin_count >= checkerboard_min_chro_bins
    ]
    return {
        "resolution": int(resolution),
        "raw_bins": int(n_bins),
        "raw_chroms": int(len(chrom_names)),
        "kept_chroms_after_zero_name": int(len(kept_chroms)),
        "max_bins_after_low_coverage": int(max(post_low_coverage_counts.values(), default=0)),
        "normdis_passing_chroms": int(len(normdis_chroms)),
        "checkerboard_passing_chroms": int(len(checkerboard_chroms)),
        "passes_normdis": bool(len(normdis_chroms) > 0),
        "passes_checkerboard": bool(len(checkerboard_chroms) > 0),
    }


def choose_filter_passing_resolution(
    mcool_path,
    normdis_min_chro_bins=DEFAULT_NORMDIS_MIN_CHRO_BINS,
    checkerboard_min_chro_bins=DEFAULT_CHECKERBOARD_MIN_CHRO_BINS,
):
    with h5py.File(mcool_path, "r") as mcool:
        resolutions = sorted((int(res) for res in mcool["resolutions"].keys()), reverse=True)

    assessments = []
    first_normdis_resolution = None
    for resolution in resolutions:
        assessment = assess_resolution_filtering(
            mcool_path,
            resolution,
            normdis_min_chro_bins=normdis_min_chro_bins,
            checkerboard_min_chro_bins=checkerboard_min_chro_bins,
        )
        assessments.append(assessment)
        if assessment["passes_normdis"] and first_normdis_resolution is None:
            first_normdis_resolution = str(resolution)
        if assessment["passes_normdis"] and assessment["passes_checkerboard"]:
            return str(resolution), assessments

    if first_normdis_resolution is not None:
        return first_normdis_resolution, assessments

    summary = ", ".join(
        f"{item['resolution']}:max_after_filter={item['max_bins_after_low_coverage']}"
        for item in assessments
    )
    raise ValueError(
        f"No resolution passes NormDis filtering with normdis_min_chro_bins={normdis_min_chro_bins}. "
        f"Assessment: {summary}"
    )


def resolve_resolution(
    mcool_path,
    requested_resolution,
    normdis_min_chro_bins=DEFAULT_NORMDIS_MIN_CHRO_BINS,
    checkerboard_min_chro_bins=DEFAULT_CHECKERBOARD_MIN_CHRO_BINS,
):
    with h5py.File(mcool_path, "r") as mcool:
        resolutions = sorted(int(res) for res in mcool["resolutions"].keys())
    if requested_resolution == "auto":
        resolution, _ = choose_filter_passing_resolution(
            mcool_path,
            normdis_min_chro_bins=normdis_min_chro_bins,
            checkerboard_min_chro_bins=checkerboard_min_chro_bins,
        )
        return resolution
    requested = int(requested_resolution)
    if requested not in resolutions:
        raise ValueError(f"Resolution {requested} not found. Available: {resolutions}")
    return str(requested)
