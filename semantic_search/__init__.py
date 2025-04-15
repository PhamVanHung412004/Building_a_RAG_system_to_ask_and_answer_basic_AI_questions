import numpy as np
from pathlib import Path
from Input import Init_Input

class Sematic_search(Init_Input):
    def __init__(self, model, use_query: str, top_k: int) -> None:
        '''
        model : model embedding sửa dụng
        use_query : câu hỏi của người dùng nhập vao
        top_k : là k vector gần với câu hỏi của người dùng nhập vào dưới dạng vector trong không gian 384 chiều
        '''
        super().__init__(use_query, top_k)
        self.model = model

    def run(self, read_model_vectordatabse) -> list:
        index = read_model_vectordatabse
        query_embedding = self.model.encode([self.use_query])[0]
        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
        D, I = index.search(query_embedding, self.top_k)
        return I
