import os, time, json
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


def batch_rerank(results: list[dict], query: str) -> list[dict]:
        drs_list_info = []
        drs_dict = {}
        for result in results:
            drs_list_info.append(f'{result['doc']['id']} - {result['doc']['name']} - {result['doc']['specialty']} - {result['doc']['bio']} - {result['doc']['age']} - {result['doc']['availability']}')
            drs_dict[result["doc"]["id"]] = result["doc"]
        prompt = f"""
            Rank the medical providers/doctors listed below by their relevance to the specific patient inquiry.

            Query: "{query}"

            Doctors Information:
            {drs_list_info}

            Consider:
            - Precise specialty alignment (e.g., matching 'Heart' to 'Cardiology' over 'General Surgery').
            - Specificity of the clinical expertise mentioned in the doctor's bio.
            - Alignment with requested availability or specific hospital departments.
            - Mention of specific procedures or symptoms requested by the user.

            Return ONLY the Provider IDs (or indices) in order of relevance (best match first). 
            Return a valid JSON list of integers/IDs, nothing else.

            For example:
            ["DR502", "DR108", "DR334", "DR21", "DR9"]

            Ranking:"""
        response = client.models.generate_content(
        model='gemma-3-27b-it', contents=prompt)
        ranking = json.loads(response.text)
        id_to_rank = {id: i for i, id in enumerate(ranking)}
        results = sorted(results, key=lambda result: id_to_rank[result["doc"]["id"]])
        print(ranking)
        print(results)
        return results

def rerank_results(results: list[dict], query: str, method: str) -> list[dict]:
    match method:
        case "individual":
            return individual_rerank(results, query)
        case "batch":
            return batch_rerank(results, query)
        case _:
            return results
        
