from package import pd
class Read_File:
    def __init__(self, path : str) -> None:
        '''
        path : duong dan den file csv
        '''
        self.__path : str = path

    def run(self) -> pd.core.frame.DataFrame:
        return pd.read_csv(self.__path)
    