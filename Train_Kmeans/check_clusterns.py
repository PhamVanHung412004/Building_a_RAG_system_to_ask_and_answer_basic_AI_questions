from package import (
    KElbowVisualizer,
    plt,
    numpy as np,
    KMeans
)

class Check_Cluster:
    def __init__(self, data : np.array) -> None:
        '''
        data : vector embedding cua tung cau da chunking
        '''
        self.data : np.array = data
    
    def show(self) -> None:        
        model = KMeans()
        self.model = model
        show_screen = KElbowVisualizer(model, k = (1,100))
        fig = plt.figure(figsize=(10,8))
        show_screen.fit(self.data)
        show_screen.poof()
    