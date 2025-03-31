from package import np
from package import sqrt
from package import KNeighborsClassifier

def list_calc_score_TB(range_k : int, x_train : np.array, y_train : np.array, point_new : np.array) -> list:
    '''
    range_k : luu cac gia tri cua k chay tu 1 -> sqrt(N) voi n la so luong mau thu
    x_train : vector dau vao cua cac feature da duocc chon de train
    y_train : nhan di theo tung diem trong vector dau vao
    point_new : vector diem moi den can phai du doan
    '''
    scores = []
    for k in range_k:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        y_predict = model.predict(point_new)
        scores.append(y_predict)
    return scores

class Init_KNN:
    def __init__(self, feature : np.array, labels : np.array, point_new : np.array) -> None:
        '''
        feature : vector chua so luong input dau vao
        labels : nhan cua tung diem du lieu trong vector
        point_new : diem moi xuat hien
        '''
        self.__feature = feature
        self.__labels = labels
        self.__point_new = point_new

    # return value best k
    def Search_k_very_good(self) -> int:
        sqrt_n = int(sqrt(len(self.__feature)))
        range_k = range(1,sqrt_n + 1)
        print("jjjj")
        list_distances = list_calc_score_TB(range_k,self.__feature,self.__labels, self.__point_new)        
        best_k = range_k[np.argmax(list_distances)]
        return best_k
    
    # run progaming
    def run(self) -> None:
        k_best = self.Search_k_very_good()
        model = KNeighborsClassifier(n_neighbors=k_best)
        model.fit(self.__feature,self.__labels)
        distance, labels = model.predict(self.__point_new)
        return labels