import pandas as pd
import numpy as np
import sys, os
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import save_model, load_dataset
from prediction_model.processing import preprocessing as pp
from prediction_model.pipeline import classification_pipeline


def perform_training():
    
    train_data = load_dataset(config.TRAIN_PATH)
    train_y = train_data[config.TARGET].map({"N":0, "Y":1})
    classification_pipeline.fit(train_data[config.FEATURES], train_y)
    save_model(classification_pipeline)
    

if __name__=='__main__':
    perform_training()