import string
import os
import pickle
import math

from collections import Counter, defaultdict
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from lib.search_utils import load_doctors, get_stopwords, cache_path, DEFAULT_SEARCH_LIMIT

lemmatizer = WordNetLemmatizer()
# {"id": "DR210", "name": "Dr. Agatha Christie", "age": 45, "specialty": "Forensic Toxicology", "availability": "Mon-Fri 09:00-17:00", "bio": "Expert in identifying chemical agents and drug interactions in complex cases."}
class InvertedIndex:
    def __init__(self) -> None:
        self.index: defaultdict[str, set[int]] = defaultdict(set) # Tokens to doc_ids
        self.docmap: dict[int, dict] = {} # doc_ids to full_docs
        self.term_frequencies: defaultdict[int, Counter[str, int]] = defaultdict(Counter)
        self.index_path = os.path.join(cache_path, "index.pkl") 
        self.docmap_path = os.path.join(cache_path, "docmap.pkl")
        self.term_frequencies_path = os.path.join(cache_path, "term_frequencies.pkl")

    def build(self) -> None:
        doctors_data = load_doctors()
        for _, doctor in enumerate(doctors_data):
            self.docmap[doctor["id"]] = doctor
            doctor_info = f"{doctor["name"]}. {doctor["age"]}. {doctor["specialty"]}. {doctor["bio"]}. {doctor["availability"]}" 
            self.__add_document(doctor["id"], doctor_info)

    def save(self) -> None:
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.index, f)
            
            with open(self.docmap_path, "wb") as f:
                pickle.dump(self.docmap, f)

            with open(self.term_frequencies_path, "wb") as f:
                pickle.dump(self.term_frequencies, f)

            print(f"Successfully cached hospital data")

        except Exception as e:
            print(f"Error saving hospital data: {e}")

    def load(self) -> None:
        try:           
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)

            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)

            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            
            print(f"Successfully loaded hospital data")

        except Exception as e:
            print(f"Error loading hospital data: {e}")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenization(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_document(self, term: str) -> list[int]:
        ids_of_term = sorted(self.index[term])
        return ids_of_term

    def get_tf(self, doc_id: str, term: int) -> int:
        tokens = tokenization(term)
        if len(tokens) != 1:
            raise ValueError("The term has to be one word.")
        return self.term_frequencies[doc_id][tokens[0]]
    
    def get_idf(self, term: str) -> float:
        total_doctors = len(self.docmap)
        term_match_doctors = len(self.index[term])
        return math.log((total_doctors + 1) / (term_match_doctors + 1))


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[str]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenization(query)
    doc_ids = []
    results = []
    for q_token in query_tokens:
        doc_ids.extend(idx.get_document(q_token))
        if len(doc_ids) >= limit:
            for doc_id in doc_ids[:limit]:
                doctor_info = idx.docmap[doc_id]
                results.append(f"Name: {doctor_info["name"]}. ID: {doc_id}")
            break
    return results


def get_tf_command(doc_id: str, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def get_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def get_tfidf_command(doc_id: str, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)
    return tf * idf

def tokenization(text: str) -> list[str]:
    lowered_text = text.lower()
    table = str.maketrans("", "", string.punctuation)
    normalized_text = lowered_text.translate(table)
    words = normalized_text.split()
    stopwords = set(get_stopwords())

    tagged_words = pos_tag(words)
    
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_words if word and word not in stopwords]
    return tokens


def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN