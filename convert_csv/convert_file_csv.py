import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def read_file(path : str = None) -> str:
    '''
    path : đường dẫn của file .txt
    '''
    with open(path, "r", encoding="utf-8") as file:
        data = file.read()
        return data if (len(data) > 2) else "-1"

def get_value(name_context : str) -> int:
    '''
    name_context : tên của file context
    '''
    index = name_context.index(".")
    t = index - 1
    value = []
    cnt = 0
    while(name_context[t] >= "0" and name_context[t] <= "9"):
        value.insert(cnt,name_context[t])
        t -= 1
    number = ""
    for i in value:
        number += i
    return int(number)

def main():
    path = r"E:\datalaptop\D\PROJECT_GITHUB\ChatbotAIO\content"
    model = SentenceTransformer("all-MiniLM-L6-v2")  

    # Sắp xếp lại theo đúng thứ tự đã chunking
    list_arr = os.listdir(path)
    list_arr.sort(key=lambda x : get_value(x))

    vector_document = [read_file(path + "/" + list_arr[i]) for i in range(len(list_arr)) if (read_file(path + "/" + list_arr[i]) != "-1")] # Đọc từng file text rồi lưu vào danh sách

    embeddings = np.array([model.encode(doc).tolist() for doc in vector_document])  # Chuyển thành list để lưu vào CSV
    df = pd.DataFrame({
        "chunk_id": range(1, len(vector_document) + 1),
        "text": vector_document,
        "embedding": [json.dumps(emb.tolist()) for emb in embeddings],  # Chuyển embedding thành JSON để lưu vào CSV
    })

    df.to_csv("dataset.csv", index=False, encoding="utf-8")

main()


