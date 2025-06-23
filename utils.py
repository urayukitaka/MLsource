import json
import pickle
import numpy as np

# dict <-> json file
def save_jsonf(dict_data:dict,
               savefilename:str,
               encoding:str="utf-8"):
    with open(savefilename, "w", encoding=encoding) as f:
        json.dump(dict_data, f, indent=2, ensure_ascii=False)

def open_jsonf(filename:str,
               encoding:str="utf-8"):
    with open(filename, "r", encoding=encoding) as f:
        loaded_data = json.load(f)
    return loaded_data

# object <-> pkl file
def save_pkl(data,
             savefilename:str):
    with open(savefilename, "wb") as f:
        pickle.dump(data, f)

def open_pkl(filename:str):
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data

# for mdeling
def convert_numpy(obj):
    """
    NumPy型（int, float, bool, ndarrayなど）を再帰的にPythonの組み込み型へ変換
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    else:
        return obj