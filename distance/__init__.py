import numpy as np
from numpy import dot
from numpy.linalg import norm 
from numpy.typing import NDArray
class Distance:
    def __init__(self, vector1 : NDArray[np.float32], vector2 : NDArray[np.float32]) -> None:
        '''
        vector1 : là vector của điểm thứ nhất
        vector2 : là vector của điểm thứ hai
        '''
        self.__vector1 : NDArray[np.float32] = vector1
        self.__vector2 : NDArray[np.float32] = vector2

    def return_value_distance(self) -> float:
        cos_sim = dot(self.__vector1, self.__vector2) / (norm(self.__vector1) * norm(self.__vector2))
        cos_dist = 1 - cos_sim
        return cos_dist
