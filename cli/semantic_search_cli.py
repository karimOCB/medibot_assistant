#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Check model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed input text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query text")
    embed_query_parser.add_argument("query", type=str, help="Text query to embed")
    
    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", type=str, help="Text query to search")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Optional results limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embedding = embed_text(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}") # 3 first dimensions
            print(f"Dimensions: {embedding.shape[0]}") # quantity of dimensions
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            results = search_command(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. Name: {result['name']} (score: {result['score']:.4f})\n{result['dr_info']}\n")
        case _:
            parser.print_help()
            
if __name__ == "__main__":
    main()