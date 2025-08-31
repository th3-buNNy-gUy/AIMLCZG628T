import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn import metrics

warnings.filterwarnings("ignore")

SEED = 42

class CommonUtility():

    @staticmethod
    def metric_scores(cf):
        accuracy  = np.around(np.trace(cf) / float(np.sum(cf)), 4)
        precision = np.around(cf[1,1] / sum(cf[:,1]), 4)
        recall    = np.around(cf[1,1] / sum(cf[1,:]), 4)
        f1_score  = np.around(2*precision*recall / (precision + recall), 4)
        return accuracy, precision, recall, f1_score

    @staticmethod
    def format_confusion_matrix(cf):
        tp = cf[0, 0]
        fp = np.sum(cf[1:, 0])
        fn = np.sum(cf[0, 1:])
        tn = np.sum(cf[1:, 1:])
        return np.array([[tp, fn], [fp, tn]])

    @staticmethod
    def confusion_matrix(y_test, y_pred):
        return metrics.confusion_matrix(y_test, y_pred)

    @staticmethod
    def split_dataset(dataframe, anomaly_column, test_split, val_split, stratify):
        dataframe_X = dataframe.drop(columns=[anomaly_column])
        dataframe_y = dataframe[[anomaly_column]]

        X_train, X_test, y_train, y_test = train_test_split(
                                                                dataframe_X.values, 
                                                                dataframe_y.values.ravel(), 
                                                                test_size=test_split, 
                                                                random_state=SEED, 
                                                                stratify=dataframe_y.values.ravel() if stratify else None
        )
        dataframe_X_train = pd.DataFrame(X_train, columns=dataframe_X.columns)
        dataframe_X_test = pd.DataFrame(X_test, columns=dataframe_X.columns)
        dataframe_y_train = pd.DataFrame(y_train, columns=dataframe_y.columns)
        dataframe_y_test = pd.DataFrame(y_test, columns=dataframe_y.columns)

        X_val, X_test, y_val, y_test = train_test_split(
                                                                dataframe_X_test.values, 
                                                                dataframe_y_test.values.ravel(), 
                                                                test_size=val_split, 
                                                                random_state=SEED, 
                                                                stratify=dataframe_y_test.values.ravel() if stratify else None
        )
        dataframe_X_val = pd.DataFrame(X_val, columns=dataframe_X.columns)
        dataframe_X_test = pd.DataFrame(X_test, columns=dataframe_X.columns)
        dataframe_y_val = pd.DataFrame(y_val, columns=dataframe_y.columns)
        dataframe_y_test = pd.DataFrame(y_test, columns=dataframe_y.columns)

        return pd.concat([dataframe_X_train, dataframe_y_train], axis=1), pd.concat([dataframe_X_val, dataframe_y_val], axis=1), pd.concat([dataframe_X_test, dataframe_y_test], axis=1)
