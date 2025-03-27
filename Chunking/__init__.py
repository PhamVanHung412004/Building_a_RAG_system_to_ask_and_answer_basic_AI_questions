import sys
sys.path.append("/content/drive/MyDrive/LLM/chatbot/ChatbotAIO")
from Tutorial import *
class Chunking_Data:
    def __init__(self,documents : list, model_embedding : HuggingFaceEmbedding) -> None:
        '''
        documents : Văn bản sau khi đã chuyển hóa từ file PDF thành file text và được lưu dưới dạng list
        model_embedding : là model embedding do mình lựa chọn để chunking data
        '''
        self.__documents : list = documents
        self.__model_embedding = model_embedding

    def Get_nodes(self) -> list:        
        try:
            splitter = SemanticSplitterNodeParser(E
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.__model_embedding
            )
            nodes = splitter.get_nodes_from_documents(self.__documents)
            return nodes
        except ZeroDivisionError as e:
            print("Error: {}".format(e))
