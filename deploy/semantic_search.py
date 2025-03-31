from package import Init_Input
from package import np
from package import KMeans


class Sematic_search(Init_Input):
    def __init__(self,use_query : str = None, top_k : int = None) -> None:
        super().__init__(use_query,top_k)