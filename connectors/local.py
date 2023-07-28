from config.config import DATA_DIR
import pandas as pd
import streamlit as st


def display_objets_in_local(dir=DATA_DIR):
    all_files = [str(file) for file in dir.rglob("*") if file.is_file()]
    all_files = [file.split(str(dir))[1] for file in all_files]
    files = {
        "Folder": ["/".join(file.split("/")[:-1]) for file in all_files],
        "File": [file.split("/")[-1] for file in all_files],
    }
    files = pd.DataFrame(files)
    print(files.columns)
    return files


def interface(user):
    print("Displaying objects in local")
    st.table(display_objets_in_local())


if __name__ == "__main__":
    print("Displaying objects in local")
    print(display_objets_in_local())
