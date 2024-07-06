import sys, os
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
print(PACKAGE_ROOT)

from prediction_model.processing.data_handling import load_model, load_dataset
import pandas as pd
import numpy as np
from prediction_model.config import config
import joblib

classification_pipeline = load_model()

def generate_prediction(data_input):
    
    df = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(df[config.FEATURES])
    output = np.where(pred==1, "Y", "N")
    result = {"prediction":output}
    return result

# def generate_prediction():
    
#     df = load_dataset(config.TEST_PATH)
#     pred = classification_pipeline.predict(df[config.FEATURES])
#     output = np.where(pred==1, "Y", "N")
#     return output


if __name__=='__main__':
  
    generate_prediction()