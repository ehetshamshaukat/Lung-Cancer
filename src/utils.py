import os
import pickle


def save_file_as_pickle(obj_name,obj_path):
    os.makedirs(os.path.dirname(obj_path),exist_ok=True)
    with open(obj_path,"wb") as path:
        pickle.dump(obj_name,path)

def load_pickle(path):
    with open(path,"rb") as path_obj:
        return pickle.load(path_obj)

