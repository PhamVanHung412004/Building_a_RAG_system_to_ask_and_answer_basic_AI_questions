from package import Counter
from package import np

class init_Hybrid_Search:
    def __init__(self, use_query : str , top_k : int ,documents : np.array, result_keyword_search : list, result_senamtic_search : list) -> None:
        '''
        use_query : cau hoi cua nguoi dung nhap vao
        top_k : so tai lieu gan voi cau use_query cua nguoi dung nhat
        documents : vector chua cac cau da chunking
        result_keyword_search : vector chua cac tu cung keyword
        result_senamtic_search : vector chua cac tu cung ngu canh
        '''
        self.use_query = use_query
        self.top_k = top_k
        self.documents = documents
        self.result_keyword_search = result_keyword_search
        self.result_senamtic_search = result_senamtic_search
    
    def run(self) -> list:
        combined_results = self.result_keyword_search + self.result_senamtic_search
        reranked = [idx for idx, _ in Counter(combined_results).most_common()]
        reranked = [int(i) for i in reranked[:self.top_k]]  # Ép kiểu về int

        return [self.documents.iloc[i] for i in reranked]  # Sử dụng iloc nếu documents là DataFrame


