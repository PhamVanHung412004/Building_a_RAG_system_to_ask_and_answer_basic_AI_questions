from package import (
    KMeans,
    numpy as np,
    silhouette_score
)

class Build_KMeans:
    def __init__(self,data : np.array = None, number_clusters : int = None) -> None:
        '''
        data : Vector embedding sau khi chunking
        number_clusters : Số cụm muốn phân loại
        '''
        self.__data : np.array = data
        self.__number_clusters : int = number_clusters
    
    def run(self):
        model_train = KMeans(n_clusters=self.__number_clusters, random_state=42)
        return model_train.fit(self.__data)

    def test(self) -> np.array:
        return self.run().labels_

    def get_labels(self) -> np.array:
        return self.run().predict(self.__data)

    def get_center_point(self) -> np.array:
        return self.run().cluster_centers_

    def feeback(self) -> float:
        return silhouette_score(self.__data,self.get_labels())
