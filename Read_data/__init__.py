import pandas as pd
from pathlib import Path

class Get_Path:
    def __init__(self, path : str) -> None:
        self.path = path

class Read_File_CSV(Get_Path):
    def __init__(self, path : str) -> None:
        Get_Path.__init__(self,path)

    def Read(self) -> pd.core.frame.DataFrame:
        return pd.read_csv(self.path)

class Read_File_PDF(Get_Path):
    ...
