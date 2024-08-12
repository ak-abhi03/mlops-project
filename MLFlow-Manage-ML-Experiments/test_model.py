import mlflow
import pandas as pd
logged_model = 'runs:/80410e1ccd814befaedcea8482c0844d/RandomForestClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.

data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")

