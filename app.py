print("Pham Van Hung")
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from semantic_search import Sematic_search
from read_file import (
    Read_File_CSV,
    pd
)
from langdetect import detect,LangDetectException
from deep_translator import GoogleTranslator
file_path_dataset_file_csv = Path(__file__).parent / "dataset.csv" 
file_path_dataset_file_vector_database = Path(__file__).parent / "save_vector_and_file_json" / "vector_database.faiss" 
file_path_json = Path(__file__).parent / "save_vector_and_file_json" / "clusters_points.json" 
# print("Pham Van Hung")
# # load model


@st.cache_resource
def load_model() -> tuple:
    DATASET_TEXT = Read_File_CSV(file_path_dataset_file_csv).run()["text"]
    VECTOR_DATABASE = faiss.read_index(str(file_path_dataset_file_vector_database))
    MODEL_EMBEDDING = SentenceTransformer("BAAI/bge-small-en-v1.5")
    DATA_jSON = Read_File_JSON(file_path_json)
    return (MODEL_EMBEDDING,DATASET_TEXT,VECTOR_DATABASE, DATA_jSON)


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
                    dataset : pd.core.series.Series,
                    vector_database : faiss.swigfaiss_avx2.IndexFlatL2,
                    use_query : str):
    search_sematic = Sematic_search(model,use_query,1)
    labels_neghir = search_sematic.run(vector_database)[0][0]
    print(labels_neghir)
    
    pass
        
def is_vietnamese(text):
    if not text or len(text.strip()) < 5:  # có thể điều chỉnh ngưỡng tuỳ trường hợp
        return False
    try:
        return detect(text) == 'vi'
    except LangDetectException:
        return False
        
# User-provided prompt
def check_input_user(model, dataset,vector_database):
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
                        response = generate_response(model, dataset,vector_database,prompt)
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)
                # with st.chat_message("assistant"):
                #     text_to_speech(response)
                #     if os.path.exists("response.mp3"):
                #         if st.button("▶️ Nghe câu trả lời"):
                #             st.audio("response.mp3", format="audio/mp3")
                            
        except ZeroDivisionError as e:

            title = e
            if detect(title) != 'vi':
                answer = GoogleTranslator(source='auto', target='vi').translate(title)

            with st.chat_message("assistant"):    
                st.write("Lỗi: {}".format(title))
            # with st.chat_message("assistant"):
            #     st.write("Chuyển hóa văn bản thành giọng nói?")
            #     text_to_speech(response)



def main(): 
    model,dataset,vector_database, data_json = load_model()
    print(type(model))
    print(type(dataset))
    print(type(vector_database))
    check_input_user(model,dataset,vector_database)
main()

