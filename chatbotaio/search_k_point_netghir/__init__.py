import numpy as np
from numpy.typing import NDArray
from typing import Dict

class Search:
    def __init__(self, vector : list[list[float , int]], top_k : int, data_text : NDArray[np.str_]) -> None:
        '''
        vector : distance and index in vector data after
        top_k : points neighrs with vector new
        data_text : dataset type text chunking
        '''
        self.__vector : list[list[float , int]] = vector
        self.__top_k : int = top_k
        self.__data_text = data_text

    def run(self) -> list:
        sort_dict_key = self.__vector
        sort_dict_key.sort()
        return [ distace_and_index[1] for distace_and_index in sort_dict_key[ : 3]]

    def get_text(self) -> None: 
        vector_index = self.run()
        return [self.__data_text[vector_index[i]] for i in range(len(vector_index))]