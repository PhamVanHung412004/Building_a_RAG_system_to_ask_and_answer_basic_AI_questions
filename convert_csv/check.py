import numpy as np
import json
import pandas as pd

# 1️⃣ Đọc file CSV đã lưu embedding
df = pd.read_csv("dataset.csv")

# 2️⃣ Chuyển embedding từ JSON về list
df["embedding"] = df["embedding"].apply(json.loads)
# print(len(list(df[0])))
# 3️⃣ Chuyển thành mảng numpy
embeddings = np.vstack(df["embedding"].values)
print(len(list(embeddings[4])))
# 4️⃣ In số chiều của vector embedding
# print("Số chiều của vector embedding là:", embeddings.shape[1])
