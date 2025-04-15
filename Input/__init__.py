class Init_Input:
    def __init__(self, use_query : str = None, top_k : int = None) -> None:
        '''
        use_query : Câu hỏi của người dùng nhập vào
        top_k : là k điểm gần với câu trả lời nhất trong các vector embedding đã lưu từ trước
        '''
        self.use_query : str = use_query
        self.top_k : int = top_k