import os, json
from dotenv import load_dotenv
from google import genai

from .search_utils import load_golden_dataset, load_doctors
from .hybrid_search import HybridSearch
from .semantic_search import SemanticSearch


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)

def evaluation_command(limit: int) -> list[dict]:
    drs_docs = load_doctors()
    test_cases = load_golden_dataset() 

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(drs_docs)
    hybryd_search = HybridSearch(drs_docs)   

    for i, case in enumerate(test_cases):
        query = case["query"]
        relevant_docs_ids = [doc["id"] for doc in case["relevant_docs"]]
        relevant_docs_names = [doc["name"] for doc in case["relevant_docs"]]
        relevant_retrieved = 0
        results = hybryd_search.rrf_search(query, k=60, limit=limit)
        retrieved = []
        for result in results:
            if result[1]["doc"]["id"] in relevant_docs_ids:
                relevant_retrieved += 1
            retrieved.append(result[1]["doc"]["name"])
        precision = relevant_retrieved / len(results) if len(results) > 0 else 0
        recall = relevant_retrieved / len(relevant_docs_ids)
        f1 = 2 * (precision * recall) / (precision + recall)
        test_cases[i]["precision"] = precision
        test_cases[i]["recall"] = recall
        test_cases[i]["f1"] = f1    
        test_cases[i]["retrieved"] = retrieved
        test_cases[i]["relevant"] = relevant_docs_names

    return test_cases

def llm_evaluation_command(query: str, formatted_results: list[str]) -> list[str]:
    prompt= f"""Rate how relevant each result is to this query on a 0-3 scale:

            Query: "{query}"

            Results:
            {chr(10).join(formatted_results)}

            Scale:
            - 3: Highly relevant
            - 2: Relevant
            - 1: Marginally relevant
            - 0: Not relevant

            Do NOT give any numbers other than 0, 1, 2, or 3.

            Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

            [2, 0, 3, 2, 0, 1]"""
    
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    scores = json.loads(response.text)
    for i, f_r in enumerate(formatted_results):
        formatted_results[i] = f_r + "  " + "Score: " + str(scores[i])
    return formatted_results