import os
import pickle


def save_file_as_pickle(name,path):
    dir_name=os.path.dirname(path)
    os.makedirs(dir_name,exist_ok=True)
    with open(path,"wb") as path:
        pickle.dump(name,path)
