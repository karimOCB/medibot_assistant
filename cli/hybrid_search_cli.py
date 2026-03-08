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
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method",)
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Result rerank method")

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
            limit = args.limit * 5 if args.rerank_method else args.limit
                
            response = rrf_search_command(args.query, limit, args.k, args.enhance, args.rerank_method)
            if response["enhanced_query"]:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{response["enhanced_query"]}'\n")
            if args.rerank_method:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")
            print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")

            for i, result in enumerate(response["results"][:args.limit], start=1):
                print(f"{i}. {result["doc"]["name"]}")
                if "individual_score" in result:
                    print(f"   Re-rank Score: {result.get('individual_score', 0):.3f}/10")
                if "batch_rank" in result:
                    print(f"   Re-rank Rank: {i}")
                if "crossencoder_score" in result:
                    print(f"   Cross Encoder Score: {result.get('crossencoder_score', 0):.3f}")
                print(f"    RRF Score: {result["rrf_score"]:.3f}")
                print(f"    BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]} \n")
            
       
        case _: 
            parser.print_help()


if __name__ == "__main__":
    main()