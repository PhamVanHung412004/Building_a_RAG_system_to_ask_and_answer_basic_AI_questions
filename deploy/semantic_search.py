from package import Init_Input
from package import np
from package import KMeans
from package import Path
from package import faiss
from package import SentenceTransformer
from package import Path
from package import joblib

def read_model() -> dict:
    file_path_models = Path(__file__).parent / "vector_save_models"
    file_path_labels = Path(__file__).parent / "vector_save_labels"

    faiss_indices = {}
    for cluster_id in range(15):
        index = faiss.read_index(f"{file_path_models}/faiss_cluster_{cluster_id}.index")
        doc_idxs = np.load(f"{file_path_labels}/faiss_cluster_{cluster_id}_docs.npy")
        faiss_indices[cluster_id] = (index, doc_idxs)
    return faiss_indices

class Sematic_search(Init_Input):
    def __init__(self,model : SentenceTransformer, use_query : str, top_k : int) -> None:
        super().__init__(use_query,top_k)
        self.model = model

    def run(self) -> list:
        faiss_indices = read_model()
        query_embedding = self.model.encode([self.use_query])[0]
        faiss.normalize_L2(np.array([query_embedding]))
        
        file_tmp = Path(__file__).parent / "weight" / "model_KMeans.pkl" 
        kmeans_model = joblib.load(file_tmp)

        cluster_id = kmeans_model.predict([query_embedding])[0]

        if cluster_id not in faiss_indices:
            return []

        index, doc_idxs = faiss_indices[cluster_id]
        _, nearest_idx = index.search(np.array([query_embedding]), self.top_k)

        return [doc_idxs[i] for i in nearest_idx[0] if i < len(doc_idxs)]