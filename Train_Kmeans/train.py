from read_file import Read_File
from read_file import pd
from convert_embedding import Embedding_To_Numpy
from Kmeans import Kmeans
from check_clusterns import Check_Cluster

'''
python3 -m venv myenv
'''
def main():
    # Read file csv
    data = Read_File("/home/phamvanhung/Project_Github/ChatbotAIO/convert_csv/dataset.csv").run()
    data_embedding = Embedding_To_Numpy(data["embedding"]).convert_to_numpy()
    
    # check clustert good
    # check = Check_Cluster(data_embedding).show()
    
    #check score train model
    for k in [4,15,22,25]:
        train_kmeans = Kmeans(data_embedding,k)
        print("-" * 50)
        print("k: {}".format(k))
        print("score : {}".format(train_kmeans.feeback()))

    #add row name labels and labels 
    data = pd.DataFrame(data)
    data["Labels"] = Kmeans(data_embedding,15).get_labels()

    data.to_csv("dataset_train_kmeans.csv", index=False, encoding="utf-8")
main()  