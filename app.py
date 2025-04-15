import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from semantic_search import Sematic_search
from read_file import (
    Read_File_CSV,
    Read_File_JSON,
    pd,
    np
)
from langdetect import detect,LangDetectException
from deep_translator import GoogleTranslator
from typing import Dict
from numpy.typing import NDArray
from distance import Distance
from search_k_point_netghir import Search

file_path_dataset_file_csv = Path(__file__).parent / "dataset.csv" 
file_path_dataset_file_vector_database = Path(__file__).parent / "save_vector_and_file_json" / "vector_database.faiss" 
file_path_json = Path(__file__).parent / "save_vector_and_file_json" / "clusters_points.json" 


@st.cache_resource
def load_model() -> tuple:
    DATASET_TEXT = Read_File_CSV(file_path_dataset_file_csv).run()["text"].to_numpy()
    VECTOR_DATABASE = faiss.read_index(str(file_path_dataset_file_vector_database))
    MODEL_EMBEDDING = SentenceTransformer("BAAI/bge-small-en-v1.5")
    DATA_jSON = Read_File_JSON(file_path_json).run()
    return (MODEL_EMBEDDING,DATASET_TEXT,VECTOR_DATABASE, DATA_jSON)

def normalize_data(vector : NDArray[np.float32]) -> NDArray[np.float32]:
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector)
    else:
        return vector / np.linalg.norm(vector, axis=1, keepdims=True)


list_chat_history = []

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! bạn có thắc mắc gì về AI?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Sidebar to display chat history
with st.sidebar:
    st.header("Lịch sử trò chuyện")
    for message in st.session_state.messages:
        if (message['role'].capitalize() == "User"):
            title = "Câu hỏi: " + message['content'] 
            st.write(title)

# Function for generating LLM response
def generate_response(model : SentenceTransformer, 
                    dataset : NDArray[np.str_],
                    vector_database : faiss.swigfaiss_avx2.IndexFlatL2,
                    data_json : Dict[str , NDArray[np.float32]],
                    use_query : str):
    '''
    model : model embedding lựa chọn
    dataset : vector các đoạn text sau khi chunking
    vector_database : chứa các vector embedding sau khi đã chungking
    data_json : labels đi theo các điểm thuộc labels
    use_query : câu hỏi của người dùng
    '''

    search_sematic = Sematic_search(model,use_query,3)
    labels_neghir = search_sematic.run(vector_database)[0]
    for index in labels_neghir:
        print(dataset[index])
    # vector_tmp1 = model.encode(use_query,normalize_embeddings=True)
    # # print(vector_tmp1)
    # vector_tmp2 = data_json[str(labels_neghir)]

    # vector2 = np.array(vector_tmp2, dtype="float32")

    # vector_normalize2 = normalize_data(vector2)

    # vector_distance_index = [[float(Distance(vector_tmp1,vector_normalize2[i]).return_value_distance()), i] for i in range(len(vector_normalize2))]
    # vector_text_sematic = Search(vector_distance_index,1, dataset).get_text()
    # print(vector_text_sematic)

def is_vietnamese(text):
    if not text or len(text.strip()) < 5:  # có thể điều chỉnh ngưỡng tuỳ trường hợp
        return False
    try:
        return detect(text) == 'vi'
    except LangDetectException:
        return False
        
# User-provided prompt
def check_input_user(model, dataset,vector_database, data_json : Dict[str , NDArray[np.float32]]):
    if prompt := st.chat_input():
        try:
            prompt = prompt.replace(".", "?")
            if (prompt[-1] != '?'):
                if (prompt[-1] == '.'):
                    prompt[-1] = '?'
                else:
                    prompt += '?'
        
            st.session_state.messages.append({"role": "user", "content": prompt})
        
            with st.chat_message("user"):
                st.write(prompt)
        
            # Generate a response if last message is not from assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_response(model, dataset,vector_database, data_json, prompt)
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)

        except ZeroDivisionError as e:
            title = e
            if detect(title) != 'vi':
                answer = GoogleTranslator(source='auto', target='vi').translate(title)

            with st.chat_message("assistant"):    
                st.write("Lỗi: {}".format(title))

def main(): 
    model,dataset,vector_database, data_json = load_model()
    check_input_user(model,dataset,vector_database, data_json)
main()

