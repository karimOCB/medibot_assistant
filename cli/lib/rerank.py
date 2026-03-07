import os, time
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)

def individual_rerank(results: list[dict], query: str) -> list[dict]:
    for result in results:
        prompt = f"""
                Rate how well this medical provider's information matches the user's specific inquiry.

                Query: "{query}"
                Provider Info: {result["doc"].get("name", "")} - {result["doc"].get("specialty", "")} - {result["doc"].get("bio", "")} - {result["doc"].get("age", "")} - {result["doc"].get("availability", "")}

                Consider:
                - Precise specialty match (e.g., does "Cardiology" match "Heart"?)
                - Clinical relevance to the symptoms or procedures mentioned.
                - Availability or scheduling alignment if mentioned in the query.
                - Entity match (if the user asked for a specific doctor by name).

                Rate 0-10 (10 = perfect match, 0 = completely irrelevant).
                Output ONLY the number in your response, no other text or explanation.

                Score:
                """
        response = client.models.generate_content(
        model='gemma-3-27b-it', contents=prompt)
        score = (response.text or "").strip().strip('"')
        time.sleep(3)
        result["rerank_score"] = int(score)
    
    return sorted(results, key=lambda item: item["rerank_score"], reverse=True)

def rerank_results(results: list[dict], query: str, method: str) -> None:
    match method:
        case "individual":
            return individual_rerank(results, query)
        case _:
            return results