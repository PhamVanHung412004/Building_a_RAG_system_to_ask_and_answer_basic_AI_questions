
from pathlib import Path
from Read_data import Read_File_CSV


def main():
    path = Path(__file__).parent / "dataset.csv"
    data = Read_File_CSV(path).Read()
    print(data)
main()