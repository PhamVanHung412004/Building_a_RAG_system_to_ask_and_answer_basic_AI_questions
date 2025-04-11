import json


with open("/home/phamvanhung/Project_Github/Building_a_RAG_system_to_ask_and_answer_basic_AI_questions/save_vector_and_file_json/clusters_points.json", "r") as file:
    data = json.load(file)
    for first, second in data.items():
        print(first, len(second))