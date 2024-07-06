import os
import pandas as pd
import joblib
from prediction_model.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(file_path)
    return _data

def save_model(model):
    model_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(model, model_path)
    print(f"Model is saved under the name {config.MODEL_NAME}")
    
def load_model():
    model_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model = joblib.load(model_path)
    return model