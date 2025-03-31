from package import faiss
from package import np
from package import Path
from package import Read_File_CSV
from package import Read_File_Model
from package import Read_File_Labels
from package import Embedding_To_Numpy

def init_cluster_indices(labels : np.array) -> dict:
    cluster_indices = {i: [] for i in range(15)}
    for idx, label in enumerate(labels):
        cluster_indices[label].append(idx)
    
    return cluster_indices

def get_faiss_indices(cluster_indices : dict, doc_embeddings : np.array) -> dict:
    faiss_indices = {}
    for cluster_id, doc_idxs in cluster_indices.items():
        cluster_vectors = np.array([doc_embeddings[i] for i in doc_idxs])

        # convert embedding
        cluster_vectors = np.array(cluster_vectors, dtype=np.float32)
        faiss.normalize_L2(cluster_vectors)

        # init FAISS index
        index = faiss.IndexFlatL2(cluster_vectors.shape[1])
        index.add(cluster_vectors)
        faiss_indices[cluster_id] = (index, doc_idxs)
    return faiss_indices

# save FAISS index clusters
def save(faiss_indices : dict) -> None:
    file_path1 = Path(__file__).parent / "vector_save_models"
    file_path2 = Path(__file__).parent / "vector_save_labels"

    for cluster_id, (index, doc_idxs) in faiss_indices.items():
        faiss.write_index(index, f"{file_path1}/faiss_cluster_{cluster_id}.index")
        np.save(f"{file_path2}/faiss_cluster_{cluster_id}_docs.npy", doc_idxs)

def main():
    file_path = Path(__file__).parent
    
    data = Read_File_CSV(file_path.parent / "convert_csv" / "dataset.csv").run()
    model = Read_File_Model(file_path / "weight" / "model_KMeans.pkl").run()
    labels = np.array(Read_File_Labels(file_path / "weight" / "labels.pkl.npy").run())
    
    documnets = Embedding_To_Numpy(data["embedding"]).convert_to_numpy()
    cluster_indices = init_cluster_indices(labels) 
    faiss_indices = get_faiss_indices(cluster_indices,documnets)
    
    save(faiss_indices)
main()