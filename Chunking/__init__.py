from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
import torch


class Chunking_Data:
    def __init__(self,name_model : str,documents : list) -> None:
        '''
        name_model : Tên Model Embedding sẽ chọn để chunking
        documents : Văn bản sau khi đã chuyển hóa từ file PDF thành file text và được lưu dưới dạng list
        '''
        self.__name_model: str = name_model
        self.__documents : list = documents

    def Get_nodes(self) -> list:        
        try:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

            embed_model = HuggingFaceEmbedding(model_name=self.__name_model, device=device_type)

            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model
            )
            nodes = splitter.get_nodes_from_documents(self.__documents)
            return nodes
        except ZeroDivisionError as e:
            print("Error: {}".format(e))


