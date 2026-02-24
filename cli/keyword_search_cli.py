#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command, build_command, get_tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build Inverted Index and save to disk")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a term")
    tf_parser.add_argument("doc_id", type=str, help="ID of the doctor to find tf")
    tf_parser.add_argument("term", type=str, help="Term to get the frequency for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print(f"Building the Inverted Index")
            build_command()
            print(f"Build successful")
        case "search":
            print(f"Searching for doctor info: {args.query}")
            results = search_command(args.query)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result}")
        case "tf":
            t_frequency = get_tf_command(args.doc_id, args.term)
            print(f"The Term Frequency of Doctor ID: {args.doc_id}, is {t_frequency}.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()