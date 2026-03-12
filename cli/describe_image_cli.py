import argparse, mimetypes, os
from dotenv import load_dotenv
from google import genai
from google.genai import types


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)


    parser = argparse.ArgumentParser(description="describe image cli")
    parser.add_argument("--image", type=str, help="Image file path")
    parser.add_argument("--query", type=str, help="Text query to rewrite based on the image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read() #TODO

    prompt = f"""
            Given the included image and text query, rewrite the text query to improve search results from a medical professional database. 
            Make sure to:
            - Synthesize visual observations from the image (e.g., anatomical location, visible symptoms) with the text.
            - Focus on clinical-specific details (symptoms, severity, related medical specialties).
            - Use professional medical terminology where appropriate to improve search accuracy (e.g., 'dermatological' instead of 'skin').
            - Return only the rewritten query, without any additional commentary.
            """

    parts = [
    prompt,
    types.Part.from_bytes(data=img, mime_type=mime),
    args.query.strip(),
    ]

    response = client.models.generate_content(
    model='gemma-3-27b-it', contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()