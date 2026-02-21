from lib.search_utils import load_doctors

def search_command(query: str):
    doctors_data = load_doctors()
    processed_query = text_processing(query)
    result_names = []
    for _, doctor in enumerate(doctors_data): # {"id": "DR208", "name": "Dr. Frida Kahlo", "age": 42, "specialty": "Pain Rehabilitation", "availability": "Mon-Wed 11:00-16:00", "bio": "Expert in managing..."},
        doctor_info = f"Name: {doctor["name"]}, age: {doctor["age"]}, specialty: {doctor["specialty"]}, bio: {doctor["bio"]}, availability: {doctor["availability"]}" 
        processed_doctor_info = text_processing(doctor_info)
        if query in processed_doctor_info:
            result_names.append(doctor["name"])
    return result_names

def text_processing(text: str):
    lowercase_text = text.lower()
    return lowercase_text