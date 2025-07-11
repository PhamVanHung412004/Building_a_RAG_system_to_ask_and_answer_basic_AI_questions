# Dự án: Xây dựng Chatbot hỏi đáp các câu hỏi liên quan đến AI.
## Giới thiệu: Dự án do tôi làm nhằm mục đích giúp mọi người có thể hỏi đáp các câu hỏi về AI một cách thuận tiện hơn.
# Các công nghệ sử dụng.
Tôi sử dụng kỹ thuật RAG. 
Sử dụng KMeans để phân cụm dữ liệu.
# Tổ chức dự án
```bash
📦 chatbotaio
 ┣ 📂 Chunking
 ┣ 📂 Embedding_Retrival
 ┣ 📂 Get_datas
 ┣ 📂 Input
 ┣ 📂 Read_data
 ┣ 📂 Train_Kmeans
 ┣ 📂 Tutorial
 ┣ 📂 convert_embedding
 ┣ 📂 convert_file_txt
 ┣ 📂 distance
 ┣ 📂 gen
 ┣ 📂 read_file
 ┣ 📂 save_vector_and_file_json
 ┣ 📂 search_k_point_netghir
 ┗ 📂 semantic_search
```
# Ý tưởng sử dụng RAG.
![Ý tưởng](image/ID_RAG.png)
# Quá trình data và chuyển thành Documents.
![Quá trình data và chuyển thành Documents](image/Get_Data.png)
# Quá trình tách văn bản Embedding để đưa vào Vector database.
![Quá trình tách văn bản Embedding để đưa vào Vector database](image/Vector_Database.png)
# Quá trình Embedding để tiến hành Retrival.
![Quá trình Embedding để tiến hành Retrival](image/Embedding_Retriver.png)
# Trực quan hóa dữ liệu.
![data visualize](image/PCA_Show.png)
# Ý tưởng tối ưu cho hệ thống RAG bằng KMeans để tăng tốc độ truy vấn.
Ta phân cụm dữ liệu ở đây nhìn từ biểu đồ có thể thấy phân làm 3 cụm sẽ hợp lý.
![data visulize clusters KMeans](image/show_clusters.png)
Sau đó ta sẽ lưu các center point vào trong vector database và lưu nhãn kèm theo các điểm thuộc nhãn đấy rồi lưu vào file json.
