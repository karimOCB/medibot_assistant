import argparse
from lib.hybrid_search import normalize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="list of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            for n_s in normalized_scores:
                print(f"* {n_s:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()