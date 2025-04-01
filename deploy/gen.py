from package import T5Tokenizer
from package import T5ForConditionalGeneration

# Tải mô hình và tokenizer
model_name = "google/flan-t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
# # Câu hỏi của người dùng
# user_question = "What is hybrid search?"

# # Danh sách các văn bản trả về từ Hybrid Search
# documents = [
#     "Hybrid search is a technique that combines both keyword-based search and semantic search. This method retrieves documents that are relevant based on both the exact words and the meaning behind the terms.",
#     "Keyword-based search matches documents based on the exact terms used in the query, while semantic search understands the meaning and context behind those terms to retrieve relevant documents.",
#     "Hybrid search aims to improve search results by leveraging the strengths of both keyword search and semantic understanding to provide more comprehensive results."
# ]

# Hàm trả lời câu hỏi từ các văn bản
class Answer_Question_From_Documents:
    def __init__(self,question : str, documents : list[str]) -> None:
        self.question = question
        self.documents = documents

    def run(self) -> str:
        # Kết hợp câu hỏi và tất cả các văn bản
        string = ""
        for i in self.documents:
            string += i + " "
        input_text = f"""
            Question: {self.question}
            Context: {string}  # Xóa xuống dòng để context rõ ràng hơn
            Answer:
        """
        # input_text = f"question: {self.question}\ncontext: {}\n " + string 
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Sinh câu trả lời
        outputs = model.generate(**inputs, max_length=100, num_beams=4, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        
        # Giải mã kết quả thành câu trả lời
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

# Áp dụng hàm và in câu trả lời
# answer = answer_question_from_documents(user_question, documents)
