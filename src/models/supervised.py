import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier # Added XGBClassifier
from sklearn.ensemble import RandomForestClassifier # Added RandomForestClassifier

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42

def test_model(model, scaler, dataframe, anomaly_column, description_column, use_scaler=True):
    X_test = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_test = dataframe[anomaly_column].values.ravel()

    X_test_scaled = scaler.transform(X_test) if use_scaler else X_test
    y_pred = model.predict(X_test_scaled)
    return y_pred, y_test

def logistic_regression(dataframe, anomaly_column, description_column, solver, C, use_scaler=True):
    X_train = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_train = dataframe[anomaly_column].values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if use_scaler else X_train

    model = LogisticRegression(solver=solver, C=C) # 'lbfgs' is a good general-purpose solver
    model.fit(X_train_scaled, y_train)
    return model, scaler

def support_vector_machine_classifier(dataframe, anomaly_column, description_column, kernel, C, use_scaler=True):
    X_train = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_train = dataframe[anomaly_column].values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if use_scaler else X_train

    model = SVC(kernel=kernel, C=C)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def extreme_gradient_boosting_classifier(dataframe, anomaly_column, description_column, lr, n_estimators, use_scaler=True):
    X_train = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_train = dataframe[anomaly_column].values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if use_scaler else X_train
    
    model = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_train)), random_state=SEED, learning_rate=lr, n_estimators=n_estimators)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def random_forest_classifier(dataframe, anomaly_column, description_column, criterion, n_estimators, use_scaler=True):
    X_train = dataframe.drop(columns=[anomaly_column, description_column]).values
    y_train = dataframe[anomaly_column].values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) if use_scaler else X_train

    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=SEED)
    model.fit(X_train_scaled, y_train)
    return model, scaler