from package import np
from package import Read_File 
from package import Embedding_To_Numpy
from package import Path
from package import KNeighborsClassifier
from package import SentenceTransformer
from package import Init_KNN

def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")  
    return model

def embedding_question():
    use_query = input("Enter question: ")
    model = load_model()
    embeddings = np.array([[model.encode(use_query).tolist()]]) # Chuyển thành vector embedding
    return embeddings

def main():
    path = Path(__file__).parent / "Train_Kmeans" / "dataset_train_kmeans.csv"
    # target_dir = path.parent / "Train_Kmeans" / "dataset_train_kmeans.csv"
    datas = Read_File(path).run()
    data_embeddings = Embedding_To_Numpy(datas["embedding"]).convert_to_numpy()    
    print(data_embeddings)
    # get_labels = np.array(datas["Labels"])
    # point_new = embedding_question()
    # labels_check = Init_KNN(data_embeddings,get_labels,point_new).run()
main()