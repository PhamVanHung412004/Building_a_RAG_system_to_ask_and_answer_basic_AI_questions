from package import keyword_search
from package import Sematic_search
from package import init_Hybrid_Search
from package import Read_File_CSV
from package import Path
from package import BM25Okapi
from package import SentenceTransformer
from package import np
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")  
    return model
def main():
    file_path = Path(__file__).parent
    file_taget = file_path.parent / "convert_csv" / "dataset.csv"
    documents = Read_File_CSV(file_taget).run()

    tokenized_docs = [doc.lower().split() for doc in documents["text"]]
    bm25 = BM25Okapi(tokenized_docs)

    use_query = input("Enter Quesition: ")
    
    result_keyword_search = keyword_search(use_query,3,bm25).run()
    vector1 = [int(x) for x in result_keyword_search]
    
    result_semantic_search = Sematic_search(load_model(),use_query,3).run()
    vector2 = [int(x) for x in result_semantic_search]
    
    Hybird_search = init_Hybrid_Search(use_query,3,documents,result_keyword_search,result_semantic_search).run()
    print(Hybird_search[0])
    # print(result_semantic_search)
main()
