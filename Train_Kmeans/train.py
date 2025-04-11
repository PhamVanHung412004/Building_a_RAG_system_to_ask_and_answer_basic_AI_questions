from package import (
    numpy as np,
    pandas as pd,
    Read_File,
    Embedding_To_Numpy,
    Path,
    Check_Cluster,
    joblib,
    Init_KMeans_FAISS,
    Build_KMeans,
    json,
    faiss
    
)



def convert_ndarray_to_list(obj):
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, (np.integer,)) else k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
    

def main():
    file_path = Path(__file__).parent.parent
    data = Read_File(file_path /  "dataset.csv").run()
    data_embedding = Embedding_To_Numpy(data["embedding"]).convert_to_numpy()

    '''
    check clustert good
    check = Check_Cluster(data_embedding).show()

    check score train model
    for k in range(2,31):
        train_kmeans = Build_KMeans(data_embedding,k)
        print("-" * 50)
        print("k: {}".format(k))
        print("score : {}".format(train_kmeans.feeback()))
    '''

    model_KMeans = Build_KMeans(data_embedding,16)

    data_labels = model_KMeans.get_labels()
    clusters_points = {}
    set_data_labels = set(data_labels)
    for i in set_data_labels:
        clusters_points[int(i)] = []        

    for i in range(len(data_labels)):
        clusters_points[int(data_labels[i])].append(data_embedding[i])

    clusters_points_new = convert_ndarray_to_list(clusters_points)
    file_path = Path(__file__).parent.parent / "save_vector_and_file_json"

    with open(file_path / "clusters_points.json", "w") as f:
        json.dump(clusters_points_new, f, ensure_ascii=False, indent=4)
    
    datas_center_new = model_KMeans.get_center_point()

    d = 384
    index = faiss.IndexFlatL2(d)
    index.add(datas_center_new)

    faiss.write_index(index, str(file_path / "vector_database.faiss"))

main()  