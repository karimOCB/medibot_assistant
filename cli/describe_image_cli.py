import argparse, mimetypes, os
from dotenv import load_dotenv
from google import genai


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
        f.read() #TODO

    promt = f"""
            Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary
            """

    #TODO