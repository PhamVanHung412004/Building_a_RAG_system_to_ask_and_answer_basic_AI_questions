from package import KMeans
from package import np
from package import silhouette_score

class Kmeans:
    def __init__(self,data : np.array = None, number_clusters : int = None) -> None:
        '''
        data : vector embedding sau khi chunking
        number_clusters : so cum muon phan
        '''
        self.__data : np.array = data
        self.__number_clusters : int = number_clusters
    
    def run(self):
        model_train = KMeans(n_clusters=self.__number_clusters, random_state=42)
        return model_train.fit(self.__data)

    def get_labels(self) -> np.array:
        return self.run().labels_

    def feeback(self) -> float:
        return silhouette_score(self.__data,self.get_labels())
