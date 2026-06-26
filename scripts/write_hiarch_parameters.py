import argparse
from pathlib import Path

from hiarch_defaults import PYTHON, SCRIPT_DIR


def as_posix_text(value):
    return Path(value).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="Output/parameters.txt")
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--script-dir", default=SCRIPT_DIR.as_posix())
    parser.add_argument("--basepath", default="Output")
    parser.add_argument("--anchor-method", default="inter_rowsum")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                "###########",
                "# Script path",
                f'python_path="{args.python}"',
                f'torch_path="{args.python}" # Only used in calculating global folding score',
                f'script_dir="{as_posix_text(args.script_dir)}"',
                "",
                "###########",
                "# Data path",
                f'basepath="{as_posix_text(args.basepath)}"',
                "",
                "###########",
                "# Parameters",
                'normdis_df="True"',
                'normdis_cmd="True"',
                'correctmap_drc="True"',
                'correctmap_ac=""',
                "checkboard_sd=0.15",
                f'GF_S1_am="{args.anchor_method}"',
                'GF_S1_ac=""',
                'GF_S1_ue="True"',
                "",
            ]
        ),
        encoding="ascii",
    )
    print(output)


if __name__ == "__main__":
    main()
