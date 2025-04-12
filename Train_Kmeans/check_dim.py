from convert_embedding import Embedding_To_Numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from read_file import Read_File
from package import(
    numpy as np,
    pandas as pd,
    Path
)

from sklearn.preprocessing import StandardScaler
def main():
    file_path = Path(__file__).parent.parent / "dataset.csv"
    file_path_save_image = Path(__file__).parent.parent / "image"

    datas = Read_File(file_path).run()

    get_datas_embedding = Embedding_To_Numpy(datas["embedding"]).convert_to_numpy()

    data_news = StandardScaler().fit_transform(get_datas_embedding)

    pca = PCA(n_components=2)
    data_PCA = pca.fit_transform(data_news)

    data_x = data_PCA[ :, 0]
    data_y = data_PCA[ :, 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(data_x, data_y, alpha=0.7, s=60, c='skyblue', edgecolors='k')
    plt.title("Biểu diễn Embedding bằng PCA (2D)")
    plt.xlabel("Thành phần chính 1")
    plt.ylabel("Thành phần chính 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path_save_image / "PCA_Show.png", dpi=300)
    plt.show()


main()