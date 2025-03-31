from package import keyword_search
from package import Read_File
from package import Path
from package import BM25Okapi
from package import SentenceTransformer
def load_model():
    return 

def main():
    # Read file csv
    path = Path(__file__).parent
    target_dir = path.parent / "deploy" / "dataset_train_KMeans.csv"
    datas = Read_File(target_dir).run()

    documents = datas["text"]
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    check = keyword_search("Hung hung",3,bm25).run()
    print(check)
main()