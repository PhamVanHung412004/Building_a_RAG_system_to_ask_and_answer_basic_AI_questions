# 🔹 Tải lại FAISS index từng cụm
faiss_indices = {}
for cluster_id in range(num_clusters):
    index = faiss.read_index(f"faiss_cluster_{cluster_id}.index")
    doc_idxs = np.load(f"faiss_cluster_{cluster_id}_docs.npy")
    faiss_indices[cluster_id] = (index, doc_idxs)
