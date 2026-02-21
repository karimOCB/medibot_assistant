from lib.search_utils import load_doctors

def search_command(query):
    doctors_data = load_doctors()
    result_names = []
    for _, doctor in enumerate(doctors_data):
        if query in doctor["name"]:
            result_names.append(doctor["name"])
    return result_names