import numpy as np
from numpy.typing import NDArray
class Distance:
    def __init__(self, vector1 : NDArray[np.float32], vector2 : NDArray[np.float32]) -> None:
        self.__vector1 : np.array = vector1
        self.__vector2 : np.array = vector2

    def return_value_distance(self) -> float:
        return np.linalg.norm(self.__vector1 - self.__vector2)
