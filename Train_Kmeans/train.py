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
    faiss,
    plt,
    silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import seaborn as sns

from sklearn.preprocessing import StandardScaler

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

    # data_new = StandardScaler().fit_transform(data_embedding)

    # Draw eblow
    # check = Check_Cluster(data_new).show()


    # build kmeans
    train_kmeans = Build_KMeans(data_embedding,3)
    data_labels = train_kmeans.get_labels()
    '''
    pcd2d = PCA(n_components=2)
    data_tmp = pcd2d.fit_transform(data_embedding)

    data_x = data_tmp[ :, 0]
    data_y = data_tmp[ :, 1]
    # Bước 6: Vẽ biểu đồ scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(data_x, data_y, alpha=0.7, s=60, c='skyblue', edgecolors='k')
    plt.title("Biểu diễn Embedding bằng PCA (2D)")
    plt.xlabel("Thành phần chính 1")
    plt.ylabel("Thành phần chính 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # check value k        
    '''

    '''
    check score train model
    for k in range(2,31):
        train_kmeans = Build_KMeans(data_embedding,k)
        print("-" * 50)
        print("k: {}".format(k))
        print("score : {}".format(train_kmeans.feeback()))
    '''

    # Write clusters and points , save vector database 
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
    
    datas_center_new = train_kmeans.get_center_point()

    d = 384
    index = faiss.IndexFlatL2(d)
    index.add(datas_center_new)

    faiss.write_index(index, str(file_path / "vector_database.faiss"))
    

    # show image
    '''
    file_path_save_result = Path(__file__).parent.parent / "image"
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(data_x, data_y, c=data_labels, cmap='tab20', s=60, alpha=0.8)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("Hiển thị cụm KMeans (k=3) trên Embedding gốc (giảm chiều bằng PCA)", fontsize=14)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    # plt.savefig(file_path_save_result / "show_clusters.png", dpi=300)
    plt.tight_layout()
    plt.show()
    '''


main()      