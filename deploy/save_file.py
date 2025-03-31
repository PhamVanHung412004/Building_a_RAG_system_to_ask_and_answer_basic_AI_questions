import faiss
import joblib
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
        faiss.normalize_L2(cluster_vectors)

        # init FAISS index
        index = faiss.IndexFlatL2(cluster_vectors.shape[1])
        index.add(cluster_vectors)
        faiss_indices[cluster_id] = (index, doc_idxs)
    return faiss_indices

# save FAISS index clusters

# def save(faiss_indices : dict) -> None:
#     file_path = Path(__file__).parent / "deploy" / "vector_database"
#     for cluster_id, (index, doc_idxs) in faiss_indices.items():
#         faiss.write_index(index, f"{file_path}/faiss_cluster_{cluster_id}.index")
#         np.save(f"faiss_cluster_{cluster_id}_docs.npy", doc_idxs)

# def test() -> None:

def main():
    file_path = Path(__file__).parent
    # print(file_path / "weight" / "model_KMeans.pkl")
    model = Read_File_Model(file_path / "weight" / "model_KMeans.pkl").run()
    labels = np.array(Read_File_Labels(file_path / "weight" / "labels.pkl.npy").run())
    print(labels)
    # init_cluster_indices(labels)
    # file_taget = file_path.parent / "convert_csv" / "dataset.csv"

    # datas = Read_File_CSV(file_taget).run()
    # embedding = Embedding_To_Numpy(datas["embedding"]).convert_to_numpy()

    # test()
main()