import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.evaluation import evaluation_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to evaluate (k for precision@k, recall@k)",)

    args = parser.parse_args()
    limit = args.limit

    test_cases = evaluation_command(limit)
    print(f"k={limit}")
    for case in test_cases:
        print(f"- Query: {case["query"]}")
        print(f"- Precision@{limit}: {case["precision"]:.4f}")
        print(f"- Recall@{limit}: {case["recall"]:.4f}")
        print(f"- F1 Score: {case["f1"]:.4f}")
        print(f"- Retrieved: {", ".join(case["retrieved"])}")
        print(f"- Relevant: {", ".join(case["relevant"])}\n")

if __name__ == "__main__":
    main()