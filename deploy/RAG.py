from package import keyword_search
from package import Sematic_search
from package import init_Hybrid_Search
from package import Read_File_CSV
from package import Path
from package import BM25Okapi
from package import SentenceTransformer
from package import Answer_Question_From_Documents
import re
import unicodedata
import time

def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")  
    return model
def clean_mixed_chars(text):
    text = re.sub(r"n\d+\n\d+", " ", text)  # Xóa chuỗi kiểu "n2\n3"
    text = re.sub(r"\n+", " ", text)  # Thay xuống dòng bằng khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()  # Xóa khoảng trắng dư
    return text
def main():
    # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1-4bit"
    # # Load mô hình trên CPU (không dùng bitsandbytes)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu")
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME, device_map="cpu", offload_state_dict=True
    # )
    # # Cấu hình quantization 4-bit
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,  # Sử dụng 4-bit quantization
    #     bnb_4bit_compute_dtype=torch.float16,  # Định dạng dữ liệu
    # )

    # # Load mô hình với quantization
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config)

    file_path = Path(__file__).parent
    file_taget = file_path.parent / "convert_csv" / "dataset.csv"
    documents = Read_File_CSV(file_taget).run()

    # tokenized_docs = [doc.lower().split() for doc in documents["text"]]
    # bm25 = BM25Okapi(tokenized_docs)

    use_query = input("Enter Quesition: ")

    start = time.perf_counter()
    # Đoạn code cần đo
    # result_keyword_search = keyword_search(use_query,3,bm25).run()
    
    result_semantic_search = Sematic_search(load_model(),use_query,3).run()
    # print(result_semantic_search)
    tmp = [clean_mixed_chars(documents["text"][int(i)]) for i in result_semantic_search]
    # Hybird_search = init_Hybrid_Search(use_query,3,documents,result_keyword_search,result_semantic_search).run()
    # list_context = [clean_mixed_chars(doc["text"]) for doc in result_semantic_search]
    # print(list_context)
    answers = Answer_Question_From_Documents(use_query,tmp).run()
    print("answers : ", answers)
    # end = time.perf_counter()
    # print(f"Thời gian chạy: {end - start:.6f} giây")

main()

