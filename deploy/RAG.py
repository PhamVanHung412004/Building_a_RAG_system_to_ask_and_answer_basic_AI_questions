from package import keyword_search
from package import Sematic_search
from package import init_Hybrid_Search
from package import Read_File_CSV
from package import Path
from package import BM25Okapi
from package import SentenceTransformer
from package import np
from package import RagSequenceForGeneration
from package import torch
from package import AutoModelForCausalLM
from package import AutoTokenizer
from package import BitsAndBytesConfig
import time

def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")  
    return model

def main():
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1-4bit"
    # Load mô hình trên CPU (không dùng bitsandbytes)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="cpu", offload_state_dict=True
    )
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

    tokenized_docs = [doc.lower().split() for doc in documents["text"]]
    bm25 = BM25Okapi(tokenized_docs)

    use_query = input("Enter Quesition: ")

    start = time.perf_counter()
    # Đoạn code cần đo
    result_keyword_search = keyword_search(use_query,3,bm25).run()
    
    result_semantic_search = Sematic_search(load_model(),use_query,3).run()
    
    Hybird_search = init_Hybrid_Search(use_query,3,documents,result_keyword_search,result_semantic_search).run()
    list_context = [doc["text"] for doc in Hybird_search]

    context_str = "\n".join(list_context)
    prompt = f"Thông tin tham khảo:\n{context_str}\n\nCâu hỏi: {use_query}\nCâu trả lời:"

    # Sinh văn bản dựa trên ngữ cảnh
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_length=200)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("answers : ", response)
    end = time.perf_counter()
    print(f"Thời gian chạy: {end - start:.6f} giây")

main()

