from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

classification_pipeline = Pipeline(
    [
        ('MeanImputer', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputer', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURES_TO_MODIFY,
                                                 variable_to_add=config.FEATURES_TO_ADD)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.FEATURES_TO_ADD)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables=config.CAT_FEATURES)),
        ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScaler', MinMaxScaler()),
        ('LogisticClassifier', LogisticRegression(random_state=33))
    ]
)