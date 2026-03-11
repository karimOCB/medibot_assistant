import os
from dotenv import load_dotenv
from google import genai

from .search_utils import load_doctors, DEFAULT_SEARCH_LIMIT
from .semantic_search import SemanticSearch
from .hybrid_search import HybridSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


def rag_command(query: str) -> tuple[list[tuple[str, dict]], str]:
    drs_docs = load_doctors()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(drs_docs)
    hybryd_search = HybridSearch(drs_docs)   
    results = hybryd_search.rrf_search(query, k=60, limit=DEFAULT_SEARCH_LIMIT)
    docs = [f"{i}. Name: {r[1]["doc"]["name"]} - Age:{r[1]["doc"]["age"]}. Specialty: {r[1]["doc"]["specialty"]}. Bio: {r[1]["doc"]["bio"]}. Availability: {r[1]["doc"]["availability"]}" for i, r in enumerate(results)]
    prompt= f"""Answer the question or provide information based on the provided documents. This should be tailored to the hospital patients.

            Query: {query}

            Documents:
            {docs}

            Provide a comprehensive answer that addresses the query:"""
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return results, corrected if corrected else query