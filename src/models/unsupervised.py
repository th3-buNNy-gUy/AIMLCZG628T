import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42

def test_unsupervised_model(model, scaler, dataframe, anomaly_column, description_column):
    X_test = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_test = dataframe[[anomaly_column]] / dataframe[[anomaly_column]]
    y_test.fillna(0, inplace=True)
    y_test = np.array([0 if idx==0 else 1 for idx in dataframe[[anomaly_column]].values.ravel()])

    X_test_scaled = scaler.transform(X_test)
    y_pred = (model.predict(X_test_scaled) == -1).astype(int)
    return y_pred, y_test

def isolation_forest(dataframe, anomaly_column, description_column, contamination, n_estimators):
    X_train = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_train = dataframe[anomaly_column].values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(contamination=contamination, random_state=SEED, n_estimators=n_estimators)
    model.fit(X_train_scaled)
    return model, scaler

def one_class_support_vector_machine(dataframe, anomaly_column, description_column, kernel, nu):
    X_train = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_train = dataframe[anomaly_column].values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = OneClassSVM(nu=nu, kernel=kernel)
    model.fit(X_train_scaled)
    return model, scaler