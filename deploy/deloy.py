from package import np
from package import Read_File 
from package import Embedding_To_Numpy
from package import Path
def main():
    path = Path(__file__).parent
    target_dir = path.parent / "Train_Kmeans" / "dataset_train_kmeans.csv"
    datas = Read_File(target_dir).run()
    data_embeddings = Embedding_To_Numpy(datas["embedding"]).convert_to_numpy()
    labels = np.array(datas["Labels"])
    print(labels)
main()