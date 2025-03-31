from package import pd
from package import joblib
from package import np
class Read_File:
    def __init__(self, path : str = None) -> None:
        '''
        path : duong dan cua file csv
        '''
        self.path : str = path

class Read_File_CSV(Read_File):
    def __init__(self, path : str) -> None:
        super().__init__(path)
    def run(self) -> pd.core.frame.DataFrame:
        return pd.read_csv(self.path)

class Read_File_Model(Read_File):
    def __init__(self, path : str = None) -> None:
        super().__init__(path)
    
    def run(self):
        return joblib.load(self.path)

class Read_File_Labels(Read_File):
    def __init__(self, path : str = None) -> None:
        super().__init__(path)
    
    def run(self):
        return np.load(self.path)
