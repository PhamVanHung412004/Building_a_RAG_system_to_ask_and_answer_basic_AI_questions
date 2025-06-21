import pandas as pd
import json
from typing import Dict
import numpy as np
from numpy.typing import NDArray

class Read_File:
    def __init__(self, path : str = None) -> None:
        '''
        path : duong dan cua file
        '''
        self.path : str = path

class Read_File_CSV(Read_File):
    def __init__(self, path : str) -> None:
        '''
        path : Dường dẫn của file csv
        '''
        super().__init__(path)
    def run(self) -> pd.core.frame.DataFrame:
        return pd.read_csv(self.path)

class Read_File_JSON(Read_File):
    def __init__(self, path : str) -> None:
        '''
        path : Đường dẫn của file Json
        '''
        super().__init__(path)

    def run(self) -> Dict[str , NDArray[np.float32]]:
        with open(self.path, "r", encoding="utf-8") as file:
            return json.load(file)
    
