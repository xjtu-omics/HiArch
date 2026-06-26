import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cooler
import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HIGASHI_ROOT = ROOT / ".external" / "Higashi"
DEFAULT_DATA_DIR = ROOT / "Data"
DEFAULT_WORK_DIR = ROOT / ".higashi_dense"
DEFAULT_ENV_PYTHON = Path(r"E:\anaconda3\python.exe")


def infer_primary_chroms(chrom_names):
    primary = []
    for chrom in chrom_names:
        if chrom.startswith("chr") and "_" not in chrom:
            suffix = chrom[3:]
            if suffix.isdigit() or suffix in {"X", "Y"}:
                primary.append(chrom)
    return primary if primary else list(chrom_names)


def read_chrom_info(mcool_path):
    with h5py.File(mcool_path, "r") as handle:
        chrom_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["resolutions"]["1000000"]["chroms"]["name"][:]
        ]
        chrom_lengths = [int(v) for v in handle["resolutions"]["1000000"]["chroms"]["length"][:]]
    return chrom_names, chrom_lengths


def export_contacts(mcool_path, out_tsv, resolution, chrom_list):
    clr = cooler.Cooler(f"{mcool_path}::resolutions/{resolution}")
    bins = clr.bins()[:][["chrom", "start"]]
    pixels = clr.pixels()[:]
    row_bins = bins.iloc[pixels["bin1_id"].to_numpy()].reset_index(drop=True)
    col_bins = bins.iloc[pixels["bin2_id"].to_numpy()].reset_index(drop=True)
    table = pd.DataFrame(
        {
            "chrom1": row_bins["chrom"].astype(str).to_numpy(),
            "pos1": row_bins["start"].astype(int).to_numpy(),
            "chrom2": col_bins["chrom"].astype(str).to_numpy(),
            "pos2": col_bins["start"].astype(int).to_numpy(),
            "count": pixels["count"].astype(float).to_numpy(),
        }
    )
    table = table[table["chrom1"].isin(chrom_list) & table["chrom2"].isin(chrom_list)]
    table.to_csv(out_tsv, sep="\t", index=False)


def prepare_project(data_dir, work_dir, resolution, neighbor_num, dimensions, cpu_num):
    mcools = sorted(data_dir.glob("*.mcool"))
    if not mcools:
        raise FileNotFoundError(f"No .mcool files found in {data_dir}")

    project_dir = work_dir / f"higashi_{resolution}"
    contacts_dir = project_dir / "contacts"
    temp_dir = project_dir / "temp"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    chrom_names, chrom_lengths = read_chrom_info(mcools[0])
    primary_chroms = infer_primary_chroms(chrom_names)
    chrom_map = dict(zip(chrom_names, chrom_lengths))
    observed_intra = set()

    cytoband_path = project_dir / "cytoband.empty.bed"
    cytoband_path.write_text("", encoding="ascii")

    filelist_path = project_dir / "filelist.txt"
    manifest = []
    with filelist_path.open("w", encoding="ascii", newline="\n") as filelist:
        for mcool in mcools:
            out_tsv = contacts_dir / f"{mcool.stem}.tsv"
            export_contacts(mcool, out_tsv, resolution, primary_chroms)
            table = pd.read_table(out_tsv)
            intra = table[table["chrom1"] == table["chrom2"]]
            observed_intra.update(intra["chrom1"].astype(str).unique().tolist())
            filelist.write(f"{out_tsv.as_posix()}\n")
            manifest.append(
                {
                    "cell_name": mcool.stem,
                    "source_mcool": str(mcool),
                    "contact_tsv": str(out_tsv),
                }
            )

    chrom_list = [chrom for chrom in primary_chroms if chrom in observed_intra]

    genome_path = project_dir / "genome.chrom.sizes"
    with genome_path.open("w", encoding="ascii", newline="\n") as handle:
        for chrom in chrom_list:
            handle.write(f"{chrom}\t{chrom_map[chrom]}\n")

    config = {
        "data_dir": str(project_dir),
        "temp_dir": str(temp_dir),
        "genome_reference_path": str(genome_path),
        "cytoband_path": str(cytoband_path),
        "input_format": "higashi_v2",
        "header_included": True,
        "contact_header": ["chrom1", "pos1", "chrom2", "pos2", "count"],
        "chrom_list": chrom_list,
        "impute_list": chrom_list,
        "resolution": resolution,
        "resolution_cell": resolution,
        "embedding_name": "dense",
        "minimum_distance": resolution,
        "maximum_distance": -1,
        "local_transfer_range": 0,
        "loss_mode": "zinb",
        "dimensions": dimensions,
        "neighbor_num": min(neighbor_num, max(len(manifest) - 1, 1)),
        "cpu_num": cpu_num,
        "cpu_num_torch": cpu_num,
        "gpu_num": 0,
        "impute_no_nbr": True,
        "impute_with_nbr": True,
        "precompute_weighted_nbr": False,
        "random_walk": False,
        "embedding_epoch": 3,
        "no_nbr_epoch": 3,
        "with_nbr_epoch": 3,
        "update_num_per_training_epoch": 120,
        "update_num_per_eval_epoch": 5,
    }
    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    manifest_path = project_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return project_dir, config_path, manifest_path


def run_command(command, cwd):
    print("RUN:", " ".join(str(x) for x in command))
    subprocess.run(command, cwd=cwd, check=True)


def run_higashi(env_python, config_path):
    run_command([str(env_python), str(HIGASHI_ROOT / "higashi" / "Higashi_wrapper.py"), "-c", str(config_path)], ROOT)
    run_command([str(env_python), str(HIGASHI_ROOT / "higashi" / "Higashi2Scool.py"), "-c", str(config_path), "-n"], ROOT)


def export_dense_mcools(project_dir, manifest_path, resolution, output_dir, neighbor_num):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scool_path = project_dir / "temp" / f"nbr_{neighbor_num}_impute.scool"
    if not scool_path.exists():
        raise FileNotFoundError(f"Imputed scool not found: {scool_path}")

    for index, item in enumerate(manifest):
        cell_uri = f"{scool_path}::/cells/cell_{index}"
        clr = cooler.Cooler(cell_uri)
        bins = clr.bins()[:][["chrom", "start", "end"]]
        pixels = clr.pixels()[:]
        output_path = output_dir / f"{item['cell_name']}_dense.mcool"
        if output_path.exists():
            output_path.unlink()
        cooler.create_cooler(
            f"{output_path}::resolutions/{resolution}",
            bins=bins,
            pixels=pixels,
            dtypes={"count": "float32"},
            ordered=True,
            symmetric_upper=True,
        )
        with h5py.File(output_path, "a") as handle:
            handle.attrs["source_mcool"] = item["source_mcool"]
            handle.attrs["model"] = "Higashi"
            handle.attrs["resolution"] = int(resolution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--resolution", type=int, default=1000000)
    parser.add_argument("--neighbor-num", type=int, default=2)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--cpu-num", type=int, default=4)
    parser.add_argument("--env-python", default=str(DEFAULT_ENV_PYTHON))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--project-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    output_dir = data_dir

    if args.project_dir is not None:
        project_dir = Path(args.project_dir)
        config_path = project_dir / "config.json"
        manifest_path = project_dir / "manifest.json"
    else:
        project_dir, config_path, manifest_path = prepare_project(
            data_dir=data_dir,
            work_dir=work_dir,
            resolution=args.resolution,
            neighbor_num=args.neighbor_num,
            dimensions=args.dimensions,
            cpu_num=args.cpu_num,
        )

    if args.prepare_only:
        print(project_dir)
        return

    if not args.export_only:
        run_higashi(Path(args.env_python), config_path)

    export_dense_mcools(project_dir, manifest_path, args.resolution, output_dir, args.neighbor_num)
    print("Dense mcool export complete.")


if __name__ == "__main__":
    main()
