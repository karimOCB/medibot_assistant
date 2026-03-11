import json
import os

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
HYBRID_A = 0.5
RRF_K = 60

current_path = os.path.abspath(__file__) # abs_path of search_utils.py
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
doctors_json_path = os.path.join(project_root_path, "data", "doctors.json")
stopwords_path = os.path.join(project_root_path, "data", "stopwords.txt")
cache_path = os.path.join(project_root_path, "cache")
golden_dataset_json_path = os.path.join(project_root_path, "data", "golden_dataset.json")

def load_doctors() -> list[dict]: # Returns a list of dicts with string keys to Any values
    if not os.path.exists(doctors_json_path):
        raise FileNotFoundError(f"The doctor database was not found at: {doctors_json_path}")

    with open(doctors_json_path, "r") as f:
        data = json.load(f)

    return data


def load_golden_dataset() -> list[dict]: # Returns a list of dicts with string keys to Any values
    if not os.path.exists(golden_dataset_json_path):
        raise FileNotFoundError(f"The doctor database was not found at: {golden_dataset_json_path}")

    with open(golden_dataset_json_path, "r") as f:
        data = json.load(f)

    return data["test_cases"]


def get_stopwords() -> list[str]:
    with open(stopwords_path, "r") as f:
        stopwords_lines = f.read()
        stopwords = stopwords_lines.splitlines()
        
    return stopwords