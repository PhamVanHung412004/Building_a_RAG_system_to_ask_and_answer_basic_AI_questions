import sys
sys.path.append("/content/drive/MyDrive/LLM/chatbot/ChatbotAIO")
from Tutorial import *
class Get_Data:
    def __init__(self, path_data: str) -> None:
        self.path_data: str = path_data
        '''
        path_data : Đường dẫn đến folder chứa file pdf cần sử dụng
        '''
    def run(self) -> list:
        try:
            return SimpleDirectoryReader(self.path_data).load_data()    
        except ZeroDivisionError as e:
            print(e)

