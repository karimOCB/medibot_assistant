#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command, build_command, tf_command, idf_command, tfidf_command, bm25idf_command, bm25tf_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT, BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build Inverted Index and save to disk")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of search results to return (default: 5)")
    
    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a term")
    tf_parser.add_argument("doc_id", type=str, help="ID of the doctor to find tf")
    tf_parser.add_argument("term", type=str, help="Term to get the frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get the Inverse Document Frequency of a term")
    idf_parser.add_argument("term", type=str, help="Term to get the idf for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get tfidf of a doctor info")
    tfidf_parser.add_argument("doc_id", type=str, help="ID for tfidf")
    tfidf_parser.add_argument("term", type=str, help="term for tfidf")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get bm25idf of a doctor info")
    bm25idf_parser.add_argument("term", type=str, help="term for tfidf")   
    
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given doctor ID and term")
    bm25_tf_parser.add_argument("doc_id", type=str, help="Doctor ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("--k1", type=float, default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("--b", type=float, default=BM25_B, help="Tunable BM25 B parameter")

    args = parser.parse_args()


    match args.command:
        
        case "build":
            print(f"Building the Inverted Index")
            build_command()
            print(f"Build successful")

        case "search":
            print(f"Searching for doctor info: {args.query}")
            results = search_command(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result}")

        case "tf":
            t_frequency = tf_command(args.doc_id, args.term)
            print(f"The Term Frequency of Doctor ID: {args.doc_id}, is {t_frequency}.")

        case "idf":
            idf_frequency = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_frequency:.2f}")

        case "tfidf":
            tfidf_score = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_score:.2f}")

        case "bm25idf":
            bm25idf_score = bm25idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf_score:.2f}")

        case "bm25tf":
            bm25tf_score = bm25tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf_score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()