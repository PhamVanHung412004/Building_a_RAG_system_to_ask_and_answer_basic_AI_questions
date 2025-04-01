from package import np
from package import faiss

class Init_KMeans_FAISS:
    def __init__(self, dimen : int , number_clutesr : int, data : np.array) -> None:
        self.dimen = dimen
        self.number_clutesr = number_clutesr
        self.data = data
    
    def run(self):
        # Huấn luyện KMeans FAISS
        kmeans_faiss = faiss.Kmeans(self.dimen, self.number_clutesr, niter=10, verbose=True)
        kmeans_faiss.train(self.data)