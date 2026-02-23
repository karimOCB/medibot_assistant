import string
import os
import pickle
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from lib.search_utils import load_doctors, get_stopwords, cache_path

lemmatizer = WordNetLemmatizer()
# {"id": "DR210", "name": "Dr. Agatha Christie", "age": 45, "specialty": "Forensic Toxicology", "availability": "Mon-Fri 09:00-17:00", "bio": "Expert in identifying chemical agents and drug interactions in complex cases."}
class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {} # Tokens to doc_ids
        self.docmap: dict[int, dict] = {} # doc_ids to full_docs
        self.index_path = os.path.join(cache_path, "index.pkl") 
        self.docmap_path = os.path.join(cache_path, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenization(text)
        for token in tokens:
            if not token in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_document(self, term: str) -> list[int]:
        ids_of_term = sorted(self.index[term])
        return ids_of_term

    def build(self) -> None:
        doctors_data = load_doctors()
        for _, doctor in enumerate(doctors_data):
            self.docmap[doctor["id"]] = doctor
            doctor_info = f"{doctor["name"]}. {doctor["age"]}. {doctor["specialty"]}. {doctor["bio"]}. {doctor["availability"]}" 
            self.__add_document(doctor["id"], doctor_info)

    def save(self):
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.index, f)
            
            with open(self.docmap_path, "wb") as f:
                pickle.dump(self.docmap, f)

            print(f"Successfully cached hospital data")

        except Exception as e:
            print(f"Error saving hospital data: {e}")


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


def search_command(query: str) -> list[str]:
    doctors_data = load_doctors()
    query_tokens = tokenization(query)
    result_names = []
    for _, doctor in enumerate(doctors_data):
        doctor_info = f"{doctor["name"]}. {doctor["age"]}. {doctor["specialty"]}. {doctor["bio"]}. {doctor["availability"]}" 
        processed_doctor_info = tokenization(doctor_info)
        for token in query_tokens:
            if token in processed_doctor_info:
                result_names.append(doctor["name"])
    return result_names


def tokenization(text: str) -> list[str]:
    lowered_text = text.lower()
    table = str.maketrans("", "", string.punctuation)
    normalized_text = lowered_text.translate(table)
    words = normalized_text.split()
    stopwords = set(get_stopwords())

    tagged_words = pos_tag(words)
    
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_words if word and word not in stopwords]
    return tokens


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    term_ids = idx.get_document("heart")
    print(f"First document for token 'heart' = {term_ids[0]}")