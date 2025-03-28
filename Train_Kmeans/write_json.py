from read_file import Read_File
from convert_embedding import Embedding_To_Numpy
from package import json

def mege_data(data_embedding : list, labels : list) -> list:
    set_tmp = list(set(labels))
    set_tmp.sort()
    data_dict = {int(i) : [] for i in set_tmp}
    for i in range(len(data_embedding)):
        data_dict[labels[i]].append(data_embedding[i])
    data_last = [{key : value} for key, value in data_dict.items()]
    return data_last


def write_file_json(dict_json : list) -> None:
    with open("dataset.json", "w", encoding="utf-8") as file:
        json.dump(dict_json,file,indent=4,ensure_ascii=False)

def main(): 
    path = "/home/phamvanhung/Project_Github/ChatbotAIO/Train_Kmeans/dataset_train_kmeans.csv"
    data = Read_File(path).run()
    data_embedding = Embedding_To_Numpy(data["embedding"]).convert_to_list()
    labels = list(data["Labels"])
    dict_json = mege_data(data_embedding,labels)
    write_file_json(dict_json)
main()