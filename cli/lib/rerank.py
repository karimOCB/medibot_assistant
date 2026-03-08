import os, time, json, torch
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

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
        result["individual_score"] = int(score)
    
    return sorted(results, key=lambda item: item["individual_score"], reverse=True)


def batch_rerank(results: list[dict], query: str) -> list[dict]:
        drs_list_info = []
        drs_dict = {}
        ids_given = []
        for result in results:
            ids_given.append({result['doc']['id']})
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

            These are exactly the ids given: {ids_given}. You MUST return a JSON list containing the same {ids_given}, re-ordered by relevance.
            (Without backticks before or after, or nothing else that can interrupt the json.loads())

            For example:
            ["DR502", "DR108", "DR334", "DR21", "DR9"]

            Ranking:"""
        response = client.models.generate_content(
        model='gemma-3-27b-it', contents=prompt)
        if not response.text:
            print("Error: API returned an empty string!")
            return results
        ranking = json.loads(response.text.strip())
        id_to_rank = {id: i for i, id in enumerate(ranking)}
        results = sorted(results, key=lambda result: id_to_rank[result["doc"]["id"]])
        return results


def cross_encoder(results: list[dict], query: str) -> list[dict]:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    pairs = []
    for result in results:
        document_text = (
                            f"Doctor Name: {result['doc']['name']}. "
                            f"Specialty: {result['doc']['specialty']}. "
                            f"Professional Bio: {result['doc']['bio']}. "
                            f"Age: {result['doc']['age']}. "
                            f"Available hours: {result['doc']['availability']}."
                        )
        pairs.append([query, document_text])
    scores = cross_encoder.predict(pairs)
    for i, result in enumerate(results):
        result["crossencoder_score"] = scores[i]
    results = sorted(results, key=lambda result: result["crossencoder_score"], reverse=True)
    return results

def rerank_results(results: list[dict], query: str, method: str) -> list[dict]:
    match method:
        case "individual":
            return individual_rerank(results, query)
        case "batch":
            return batch_rerank(results, query)
        case "cross_encoder":
            return cross_encoder(results, query)
        case _:
            return results
        
