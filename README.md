# Bài toán: Xây dựng Chatbot hỏi đáp thông tin từ silde của AIO.
# Tổ chức dự án
```bash
📦 ChatbotAIO
 ┣ 📂 Chunking # Module để chunking data
    📂 __pycache__ 
    ┗ __init__.py # Khởi tạo lên các class để chunking  
 ┣ 📂 data # Folder lưu file PDF
 ┣ 📂 Embedding_Retrival # Module để Embedding -> Retrival
    📂 __pycache__ 
    ┗ __init__.py # Khởi tạo lên các class để Embedding -> Retrival
 ┣ 📂 Get_datas # Module để lấy data
    📂 __pycache__ 
    ┗ __init__.py # Khởi tạo lên các class để lấy data và chuyển thành text lưu dưới dạng danh sách
 ┣ 📂 Image # Folder chứa ảnh cho các ý tưởng
    ┗ Embedding_Retrival.png # Ảnh biểu diễn quá trình Embedding để đưa vào Vector database 
    ┗ Get_Data.png # Ảnh biểu diễn quá trình lấy data và chuyển thành Documents 
    ┗ ID_RAG.png # Ảnh biểu diễn ý tưởng của dự án 
    ┗ Vector_Datababse.png # Ảnh biểu diễn quá trình chia dữ liệu để chuyển hóa vào Vector databse
 ┣  README.md # File mô tả giới thiệu dự án 
 ┣  main.ipynb # File code trung tâm của dự án
 ┣  setup.txt # File chứa các gói package cần cài đặt trước khi chạy chương trình  
```
### Ý tưởng
![Ý tưởng](image/ID_RAG.png)
### Quá trình data và chuyển thành Documents
![Quá trình data và chuyển thành Documents](image/Get_Data.png)
### Quá trình tách văn bản Embedding để đưa vào Vector database
![Quá trình tách văn bản Embedding để đưa vào Vector database](image/Vector_Database.png)
### Quá trình Embedding để tiến hành Retrival
![Quá trình Embedding để tiến hành Retrival](image/Embedding_Retrival.png)
