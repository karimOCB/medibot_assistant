import string
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from lib.search_utils import load_doctors, get_stopwords

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
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
    for _, doctor in enumerate(doctors_data): # {"id": "DR208", "name": "Dr. Frida Kahlo", "age": 42, "specialty": "Pain Rehabilitation", "availability": "Mon-Wed 11:00-16:00", "bio": "Expert in managing..."},
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
