#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build Inverted Index and save to disk")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()