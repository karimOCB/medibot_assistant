import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


def spell_correct(query):
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

def enhance_query(query: str, method: str=None):
    match method:
        case "spell":
            corrected = spell_correct(query)
            if corrected != query:
                print(f"Enhanced query ({method}): '{query}' -> '{corrected}'\n")
            return corrected
        case _:
            return query
