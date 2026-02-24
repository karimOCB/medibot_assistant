import json
import os

DEFAULT_SEARCH_LIMIT = 5

current_path = os.path.abspath(__file__) # abs_path of search_utils.py
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
doctors_json_path = os.path.join(project_root_path, "data", "doctors.json")
stopwords_path = os.path.join(project_root_path, "data", "stopwords.txt")
cache_path = os.path.join(project_root_path, "cache")


def load_doctors() -> list[dict]: # Returns a list of dicts with string keys to Any values
    if not os.path.exists(doctors_json_path):
        raise FileNotFoundError(f"The doctor database was not found at: {doctors_json_path}")

    with open(doctors_json_path, "r") as f:
        data = json.load(f)

    return data


def get_stopwords() -> list[str]:
    with open(stopwords_path, "r") as f:
        stopwords_lines = f.read()
        stopwords = stopwords_lines.splitlines()
        
    return stopwords