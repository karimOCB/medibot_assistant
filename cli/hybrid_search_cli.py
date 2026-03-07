import argparse
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command
from lib.search_utils import HYBRID_A, DEFAULT_SEARCH_LIMIT, RRF_K

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="query for search")
    weighted_search_parser.add_argument("--alpha", type=float, default=HYBRID_A, help="modifiable alpha variable for semantic and keyword search weighting")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit of search results")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="query for search")
    rrf_search_parser.add_argument("--k", type=int, default=RRF_K, help="modifiable K parameter. Control the decline of scores of high vs low results")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="limit of search results")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell"], help="Query enhancement method",)

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            for n_s in normalized_scores:
                print(f"* {n_s:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.enhance, args.limit)
            for i, result in enumerate(results):
                print(f"{i}. {result["doc"]["name"]}\n Hybrid Score: {result["hybrid_score"]} \n BM25: {result["bm25_normalized"]}, Semantic: {result["semantic_normalized"]} \n")
        case "rrf-search":
            response = rrf_search_command(args.query, args.limit, args.k, args.enhance)
            for i, result in enumerate(response["results"]):
                print(f"{i}. {result["doc"]["name"]}\n RRF Score: {result["rrf_score"]:.4f} \n BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]} \n")
        case _: 
            parser.print_help()


if __name__ == "__main__":
    main()