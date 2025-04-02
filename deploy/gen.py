from package import AutoModelForCausalLM
from package import AutoTokenizer
from package import torch

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


class Answer_Question_From_Documents:
    def __init__(self,question : str, documents : list[str]) -> None:
        self.question = question
        self.documents = documents

    def run(self) -> str:
        # Kết hợp câu hỏi và tất cả các văn bản
        input_text = f"""
            Question: {self.question}
            Context: {"\n".join(self.documents)}  # Xóa xuống dòng để context rõ ràng hơn
            Answer:
        """
        
        # Tokenize input
        # inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = tokenizer(input_text, return_tensors="pt")  # Mã hóa đầu vào
        inputs = inputs.to(model.device) 
        # Sinh câu trả lời
        # outputs = model.generate(**inputs, max_length=100, num_beams=4, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        outputs = model.generate(
            **inputs,
            max_length=200,  # Độ dài tối đa của câu trả lời
            temperature=0.7,  # Điều chỉnh sự đa dạng của kết quả
            top_k=50,         # Lọc từ vựng (chỉ chọn từ có xác suất cao)
            top_p=0.9,        # Nucleus sampling
        )

        # Giải mã kết quả thành câu trả lời
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return answer



# Áp dụng hàm và in câu trả lời
# answer = answer_questtion_from_documents(user_question, documents)
