import argparse
from lib.augmented_generation import rag_command, summarize_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Multi-document search result summarization")
    summarize_parser.add_argument("query", type=str, help="Query for search")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results, rag_response = rag_command(query)
            print("Search Results:")
            for r in results:
                print(f"- Name: {r[1]["doc"]["name"]} - Specialty: {r[1]["doc"]["specialty"]}")
            print("\nRag Response:")
            print(rag_response)
        case "summarize":
            results, summary = summarize_command(args.query, args.limit)
            print("Search Results:")
            for r in results:
                print(f"- Name: {r[1]["doc"]["name"]} - Specialty: {r[1]["doc"]["specialty"]}")
            print("\nLLM Response:")
            print(summary)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()