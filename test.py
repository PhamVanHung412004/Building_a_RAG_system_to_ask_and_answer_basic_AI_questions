from pathlib import Path
from chatbotaio.Read_data import Read_File_CSV


def main():
    path = Path(__file__).parent / "dataset.csv"
    data = Read_File_CSV(path).Read()
    print(data)


if __name__ == "__main__":
    main()
