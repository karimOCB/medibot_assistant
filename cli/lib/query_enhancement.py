import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


def spell_correct(query: str) -> str:
    prompt= f"""Fix any spelling errors in the user-provided doctor information search query below.
            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
            Preserve punctuation and capitalization unless a change is required for a typo fix.
            If there are no spelling errors, or if you're unsure, output the original query unchanged.
            Output only the final query text, nothing else.
            User query: "{query}"
            """
    
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query

def rewrite(query: str) -> str:
    prompt = f"""
            Rewrite the user's medical or doctor inquiry to be a concise, searchable query for a vector database.

            Consider:
            - Medical Terminology: Map symptoms or informal terms to specialties (e.g., "heart doctor" -> "Cardiologist").
            - Schedule Specifics: Extract or imply time-related keywords (e.g., "after work" -> "evening hours", "weekend").
            - Entity Recognition: Focus on doctor names, department names, and clinical services.
            - Keep the rewritten query concise (under 10 words).
            - Use a Google-style search format; do not use boolean logic (AND/OR).

            Examples:
            - "who can check my skin rash on Tuesday?" -> "Dermatologist availability Tuesday skin rash"
            - "is there a heart guy working late?" -> "Cardiologist evening hours schedule"
            - "I need to see Dr. Smith for my broken arm" -> "Dr. Smith Orthopedics schedule fracture"
            - "pediatrician open on Saturday" -> "Pediatrics department Saturday hours"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.

            User query: "{query}"
            """
    
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    rewrote = (response.text or "").strip().strip('"')
    return rewrote if rewrote else query

def expansion(query: str) -> str:
    prompt = f"""
            Expand the user-provided medical or doctor search query with related clinical terms and synonyms.

            Consider:
            - Medical Synonyms: Add formal specialties and common names (e.g., "kidney" -> "renal", "skin" -> "dermatology").
            - Clinical Context: Include related symptoms, procedures, or department names.
            - Schedule Keywords: Add terms related to appointments and availability.
            - Output only the additional terms; they will be appended to the original query.

            Examples:
            - "heart doctor" -> "cardiology cardiovascular specialist surgeon clinic"
            - "stomach pain specialist" -> "gastroenterologist GI tract abdominal digestive health"
            - "doctor for kids" -> "pediatrician pediatrics childhood development neonatology"
            - "bone fracture" -> "orthopedics orthopedic surgeon skeletal trauma X-ray"

            User query: "{query}"
            """
    
    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=prompt)
    expanded = (response.text or "").strip().strip('"')
    return expanded if expanded else query 

def enhance_query(query: str, method: str=None):
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite(query)
        case "expand":
            return expansion(query)
        case _:
            return query
