from package import np
from package import pd
from package import Read_File
from package import Kmeans
from package import Embedding_To_Numpy
from package import Path
from package import joblib
from package import Km
def main():
    # Read file csv
    data = Read_File("/home/phamvanhung/Project_Github/ChatbotAIO/convert_csv/dataset.csv").run()
    data_embedding = Embedding_To_Numpy(data["embedding"]).convert_to_numpy()
    
    model = 


    # check clustert good
    # check = Check_Cluster(data_embedding).show()
    
    #check score train model
    # for k in [4,15,22,25]:
    #     train_kmeans = Kmeans(data_embedding,k)
    #     print("-" * 50)
    #     print("k: {}".format(k))
    #     print("score : {}".format(train_kmeans.feeback()))

    # #add row name labels and labels 
    # data = pd.DataFrame(data)
    # train = Kmeans(data_embedding,15)
    
    # model = train.run()
    
    # labels = train.get_labels()
    path = Path(__file__).parent
    
    
    path_save_model = path.parent / "deploy" / "weight" / "model_KMeans.pkl"
    path_save_labels = path.parent / "deploy" / "weight" / "labels.pkl"
    # data.to_csv(target_dir, index=False, encoding="utf-8")

    joblib.dump(model, path_save_model )

# 🔹 Lưu labels để sử dụng sau này
    np.save(path_save_labels, labels)
main()  