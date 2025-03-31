from package import BM25Okapi
from package import np
from package import Init_Input

class keyword_search(Init_Input):
    def __init__(self, use_query : str = None, top_k : int = None, bm25 : BM25Okapi = None) -> None:
        super().__init__(use_query,top_k)
        self.__bm25 : BM25Okapi = bm25

    def run(self) -> list[np.int64]:
        tokenized_query = self.use_query.lower().split()
        scores = self.__bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][: self.top_k]
        return list(top_n_indices)
