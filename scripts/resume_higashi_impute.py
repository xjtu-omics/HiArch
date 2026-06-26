import argparse
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
HIGASHI_DIR = ROOT / ".external" / "Higashi" / "higashi"
if str(HIGASHI_DIR) not in sys.path:
    sys.path.insert(0, str(HIGASHI_DIR))

from Higashi_wrapper import Higashi  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--from-stage",
        choices=["stage2", "impute0", "stage3", "impute2"],
        default="stage2",
    )
    args = parser.parse_args()

    higashi = Higashi(args.config)
    higashi.process_data()
    higashi.prep_model()
    if args.from_stage == "stage2":
        higashi.train_for_imputation_nbr_0()
        higashi.impute_no_nbr()
        higashi.train_for_imputation_with_nbr()
        higashi.impute_with_nbr()
    elif args.from_stage == "impute0":
        higashi.impute_no_nbr()
        higashi.train_for_imputation_with_nbr()
        higashi.impute_with_nbr()
    elif args.from_stage == "stage3":
        higashi.train_for_imputation_with_nbr()
        higashi.impute_with_nbr()
    elif args.from_stage == "impute2":
        higashi.impute_with_nbr()


if __name__ == "__main__":
    main()
