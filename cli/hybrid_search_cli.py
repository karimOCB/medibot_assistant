import argparse
from lib.hybrid_search import normalize_command, weighted_search_command
from lib.search_utils import HYBRID_A, DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="query for search")
    weighted_search_parser.add_argument("--alpha", type=float, default=HYBRID_A, help="modifiable alpha variable for semantic and keyword search weighting")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit of search results")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            for n_s in normalized_scores:
                print(f"* {n_s:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            print(results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()