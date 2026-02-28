#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Check model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed input text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")


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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()