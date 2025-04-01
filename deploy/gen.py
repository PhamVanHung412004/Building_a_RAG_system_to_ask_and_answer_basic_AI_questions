import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

# Cấu hình quantization 4-bit để tiết kiệm tài nguyên GPU
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Chọn mô hình sinh
model_id = "Viet-Mistral/Vistral-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load mô hình với quantization 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=nf4_config,
    device_map="auto"
)

print("✅ Mô hình đã load xong!")

# Tạo prompt template
system_prompt = """
You are an expert Q&A system that is trusted around the world.
Always answer the query using the provided context information, and not prior knowledge.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...'
"""

user_prompt = """
Context information is below.
---------------------
{context_str}

User Query: {query}
"""

prompt_template = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
)

# Hàm sinh văn bản dựa trên prompt
def generate_text(context_str, query):
    # Format prompt
    formatted_prompt = prompt_template.format_messages(context_str=context_str, query=query)

    # Chuyển đổi prompt thành đầu vào của mô hình
    inputs = tokenizer(str(formatted_prompt), return_tensors="pt").to("cuda")

    # Sinh văn bản
    output = model.generate(**inputs, max_length=200)

    # Decode kết quả
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Ví dụ sử dụng
context = "The Eiffel Tower is a famous landmark located in Paris, France. It was completed in 1889 and stands 330 meters tall."
query = "Where is the Eiffel Tower located?"

response = generate_text(context, query)
print("📝 AI Response:", response)
