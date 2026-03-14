import argparse
from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Multi-document search result summarization")
    summarize_parser.add_argument("query", type=str, help="Query for search")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit")

    citations_parser = subparsers.add_parser("citations", help="Multi-document search result LLM response with citations")
    citations_parser.add_argument("query", type=str, help="Query for search")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit")

    question_parser = subparsers.add_parser("question", help="Conversational question-answering for direct llm responses")
    question_parser.add_argument("question", type=str, help="Question to response")
    question_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit")

    conversation_parser = subparsers.add_parser("conversation", help="Start an interactive conversational chat session")
    conversation_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit")


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
        case "citations":
            results, answer_citations = citations_command(args.query, args.limit)
            print("Search Results:")
            for r in results:
                print(f"- Name: {r[1]["doc"]["name"]} - Specialty: {r[1]["doc"]["specialty"]}")
            print("\nLLM Answer:")
            print(answer_citations)
        case "question":
            results, answer = question_command(args.question, args.limit)
            print("Search Results:")
            for r in results:
                print(f"- Name: {r[1]["doc"]["name"]} - Specialty: {r[1]["doc"]["specialty"]}")
            print("\nLLM Answer:")
            print(answer)
        case "conversation":
            print("Medibot Assistant: Conversational Mode")
            print("Type exit or quit to end the session. \n")

            chat_history: list[dict[str, str]] = [] 
            
            while True:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    print("MediBot: Goodbye!")
                    break
                
                if not user_input:
                    continue

                results, answer = question_command(user_input, DEFAULT_SEARCH_LIMIT, chat_history)
                
                print(f"\nMediBot: {answer}\n")
                
                chat_history.append({"input": user_input, "answer": answer})

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()