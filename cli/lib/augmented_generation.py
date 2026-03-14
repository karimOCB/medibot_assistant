import os
from dotenv import load_dotenv
from google import genai

from .search_utils import load_doctors, DEFAULT_SEARCH_LIMIT
from .hybrid_search import HybridSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


def rag_command(query: str) -> tuple[list[tuple[str, dict]], str]:
    drs_docs = load_doctors()
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

def summarize_command(query: str, limit: int) -> tuple[list[tuple[str, dict]], str]:
    drs_docs = load_doctors()
    hybryd_search = HybridSearch(drs_docs)   
    results = hybryd_search.rrf_search(query, k=60, limit=DEFAULT_SEARCH_LIMIT)
    docs = [f"{i}. Name: {r[1]["doc"]["name"]} - Age:{r[1]["doc"]["age"]}. Specialty: {r[1]["doc"]["specialty"]}. Bio: {r[1]["doc"]["bio"]}. Availability: {r[1]["doc"]["availability"]}" for i, r in enumerate(results)]

    prompt= f"""
            You are a medical assistant for a hospital. Provide a helpful response to the patient's query by synthesizing 
            information from the top doctor search results provided below.
            
            The goal is to explain which doctors are most relevant to the patient's symptoms or needs and why. 
            Focus on their specific medical expertise, the conditions they treat (from their bios), and their availability.
            
            Your response must be professional, empathetic, and information-dense. Avoid medical advice; instead, 
            focus on guiding the patient to the right specialist.
            
            Query: {query}
            Search Results:{docs}
            
            Provide a comprehensive 3–4 sentence answer that summarizes the best matches and highlights 
            key differences in their specialties or schedules:
            """
    
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return results, corrected if corrected else query

def citations_command(query: str, limit: int) -> tuple[list[tuple[str, dict]], str]:
    drs_docs = load_doctors()
    hybryd_search = HybridSearch(drs_docs)   
    results = hybryd_search.rrf_search(query, k=60, limit=DEFAULT_SEARCH_LIMIT)
    docs = [f"{i}. Name: {r[1]["doc"]["name"]} - Age:{r[1]["doc"]["age"]}. Specialty: {r[1]["doc"]["specialty"]}. Bio: {r[1]["doc"]["bio"]}. Availability: {r[1]["doc"]["availability"]}" for i, r in enumerate(results)]

    prompt = f"""Answer the medical inquiry or provide specialist information based strictly on the provided doctor profiles.

            This should be tailored to patients using the Medibot Assistant at our clinic. 

            If the provided documents do not contain enough specific information to recommend a doctor for the patient's condition, state that clearly but offer the closest possible match while citing the profiles you have.

            Query: {query}

            Documents:
            {docs}

            Instructions:
            - Provide a comprehensive answer that explains why specific doctors are a good fit for the patient's needs.
            - Cite doctor names [Pedro], [Juan], etc, format when referencing their specific expertise or availability.
            - If doctors have overlapping specialties but different availability, mention these differences to give the patient options.
            - If the answer/specialty isn't found in the documents, say "I don't have enough information in our current directory."
            - Be direct, professional, and informative.

            Answer:"""
        
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return results, corrected if corrected else query

def question_command(question: str, limit: int, chat_history: list[dict[str, str]]) -> tuple[list[tuple[str, dict]], str]:
    drs_docs = load_doctors()
    hybryd_search = HybridSearch(drs_docs)   
    results = hybryd_search.rrf_search(question, k=60, limit=DEFAULT_SEARCH_LIMIT)
    docs = [f"{i}. Name: {r[1]["doc"]["name"]} - Age:{r[1]["doc"]["age"]}. Specialty: {r[1]["doc"]["specialty"]}. Bio: {r[1]["doc"]["bio"]}. Availability: {r[1]["doc"]["availability"]}" for i, r in enumerate(results)]
    history_string = "\n".join([
        f"{i}. Patient: {turn['input']}\nAssistant: {turn['answer']}" 
        for i, turn in enumerate(chat_history, 1)
    ])
    if not chat_history:
        prompt = f"""Answer the patient's question based on the doctors available in our clinic directory.

        This should be tailored to patients using the Medibot Assistant. We are a clinic helping people find the right specialist.

        Question: {question}

        Documents:
        {docs}

        Instructions:
        - Answer questions directly and concisely.
        - Be helpful and conversational, like a friendly receptionist.
        - Avoid being overly clinical/robotic, but also don't be "hype-y" or "cringe."
        - Talk like a normal person would in a helpful chat conversation.
        - If we don't have a specific specialist, just let them know honestly.

        Answer:"""
    else:
        prompt = f"""You are the Medibot Assistant, a friendly and helpful receptionist for a medical clinic. 
                Your goal is to help patients find the right specialist from our directory.
                Relevant Documents:{docs}
                
                Conversation History: {history_string}

                Current Question: {question}

                INSTRUCTIONS:
                - Use the 'Conversation History' to understand context (e.g., who 'he' or 'she' refers to).
                - Answer the 'Current Patient Question' directly and concisely based ONLY on the provided documents.
                - Maintain a conversational, professional, and empathetic tone—like a real person.
                - If the directory does not contain a suitable specialist, inform the patient honestly.
                - Do not make up doctor names or availability if they are not in the documents.

                Answer:
                """
        
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return results, corrected if corrected else question

