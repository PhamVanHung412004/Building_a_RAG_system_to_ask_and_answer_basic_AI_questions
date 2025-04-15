import os
from groq import Groq

client = Groq(api_key="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")

class Answer_Question_From_Documents:
    def __init__(self,question : str, documents : list[str]) -> None:
        '''
        question : Câu hỏi của người dùng đưa vào
        documents : danh sách các đáp án tổng hợp từ sematic search và keyword search
        '''
        self.__question : str = question
        self.__documents : list[str] = documents

    def run(self) :
        context = "\n".join(self.__documents)  # Tạo chuỗi context bên ngoài f-string
        input_text = f"""Question: {self.__question}
        Context: {context}
        Answer:"""
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", 
                 "content": "Bạn luôn phải trả lời bằng tiếng Việt, không dùng tiếng Anh."
                },
                {
                    "role": "user",
                    "content": self.__question
                },
                {
                    "role": "assistant",
                    "content": input_text
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content

