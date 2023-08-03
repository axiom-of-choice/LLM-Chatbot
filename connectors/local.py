from config import DATA_DIR
import pandas as pd
import streamlit as st


class LocalConnector:
    def __init__(self) -> None:
        pass

    def display_objets_in_local(self, dir=DATA_DIR):
        all_files = [str(file) for file in dir.rglob("*") if file.is_file()]
        all_files = [file.split(str(dir))[1] for file in all_files]
        files = {
            "Folder": ["/".join(file.split("/")[:-1]) for file in all_files],
            "File": [file.split("/")[-1] for file in all_files],
        }
        files = pd.DataFrame(files)
        return files

    def interface(self):
        print("Displaying objects in local")
        st.table(self.display_objets_in_local())

    def write(self, obj, file_name):
        with open(file_name, "w") as f:
            f.write(obj)


if __name__ == "__main__":
    print("Displaying objects in local")
    conn = LocalConnector()
    print(conn.display_objets_in_local())
