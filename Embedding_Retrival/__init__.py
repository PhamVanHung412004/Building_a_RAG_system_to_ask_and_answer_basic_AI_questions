from Tutorial import *
class Vector_Database_Qdrant:
    def __init__(self, noder : list, 
                Name_database : str, 
                URL : str, 
                API_KEY : str,
                model_embedding : HuggingFaceEmbedding 
                ) -> None:
        '''
        noder : Documents sau khi được chunking
        Name_database : Tên tự đặt cho Database của bạn
        URL : Đường dẫn đến URL của Qdrant của bạn hãy truy cập trang web để biết rõ hơn
        API_KEY : API_KEY của bạn trên Qdrant
        model_embedding : bạn khởi tạo 1 model embedding sử dụng
        '''
        self.__noder = noder
        self.__Name_database : str = Name_database
        self.__URL : str = URL
        self.__API_KEY : str = API_KEY
        self.__model_embedding = model_embedding 

    def Init_Vector_Database(self) -> VectorStoreIndex:
        client = QdrantClient(url=self.__URL, api_key=self.__API_KEY)
        # Tạo Vector Store
        vector_store = QdrantVectorStore(client=client, collection_name=self.__Name_database)

        # Tạo StorageContext
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Tạo LlamaIndex
        index = VectorStoreIndex(
            self.__noder,
            storage_context=storage_context,
            embed_model=self.__model_embedding)           
        
        return index