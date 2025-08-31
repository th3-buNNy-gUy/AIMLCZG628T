import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import io

from sklearn.metrics import accuracy_score

from utility.common_utility import CommonUtility
from models.supervised import test_model

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)

def load_models():
    if "saved_models" in st.session_state:
        return None

    with open('./src/models/saved/logistic_regression_og.pickle', 'rb') as handle:
        lr_model_data_og = pickle.load(handle)
    with open('./src/models/saved/random_forest_classifier_og.pickle', 'rb') as handle:
        rf_model_data_og = pickle.load(handle)
    with open('./src/models/saved/xgboost_classifier_og.pickle', 'rb') as handle:
        xgb_model_data_og = pickle.load(handle)
    with open('./src/models/saved/svm_classifier_og.pickle', 'rb') as handle:
        svm_model_data_og = pickle.load(handle)

    with open('./src/models/saved/logistic_regression_smote.pickle', 'rb') as handle:
        lr_model_data_smote = pickle.load(handle)
    with open('./src/models/saved/random_forest_classifier_smote.pickle', 'rb') as handle:
        rf_model_data_smote = pickle.load(handle)
    with open('./src/models/saved/xgboost_classifier_smote.pickle', 'rb') as handle:
        xgb_model_data_smote = pickle.load(handle)

    with open('./src/models/saved/logistic_regression_adasyn.pickle', 'rb') as handle:
        lr_model_data_adasyn = pickle.load(handle)
    with open('./src/models/saved/random_forest_classifier_adasyn.pickle', 'rb') as handle:
        rf_model_data_adasyn = pickle.load(handle)
    with open('./src/models/saved/xgboost_classifier_adasyn.pickle', 'rb') as handle:
        xgb_model_data_adasyn = pickle.load(handle)

    with open('./src/models/saved/logistic_regression_cwgan.pickle', 'rb') as handle:
        lr_model_data_cwgan = pickle.load(handle)
    with open('./src/models/saved/random_forest_classifier_cwgan.pickle', 'rb') as handle:
        rf_model_data_cwgan = pickle.load(handle)
    with open('./src/models/saved/xgboost_classifier_cwgan.pickle', 'rb') as handle:
        xgb_model_data_cwgan = pickle.load(handle)

    with open('./src/models/saved/logistic_regression_cwgan_reg_ya1.pickle', 'rb') as handle:
        lr_model_data_cwgan_reg_ya1 = pickle.load(handle)
    with open('./src/models/saved/random_forest_classifier_cwgan_reg_ya1.pickle', 'rb') as handle:
        rf_model_data_cwgan_reg_ya1 = pickle.load(handle)
    with open('./src/models/saved/xgboost_classifier_cwgan_reg_ya1.pickle', 'rb') as handle:
        xgb_model_data_cwgan_reg_ya1 = pickle.load(handle)
    
    models = {
                "Logistic Regression (Dissertation)": lr_model_data_og,
                "Random Forest Classifier (Dissertation)": rf_model_data_og,
                "XGBoost Classifier (Dissertation)": xgb_model_data_og,
                "SVM Classifier (Dissertation)": svm_model_data_og,

                "Logistic Regression (SMOTE) (Dissertation)": lr_model_data_smote,
                "Random Forest Classifier (SMOTE) (Dissertation)": rf_model_data_smote,
                "XGBoost Classifier (SMOTE) (Dissertation)": xgb_model_data_smote,

                "Logistic Regression (ADASYN) (Dissertation)": lr_model_data_adasyn,
                "Random Forest Classifier (ADASYN) (Dissertation)": rf_model_data_adasyn,
                "XGBoost Classifier (ADASYN) (Dissertation)": xgb_model_data_adasyn,

                "Logistic Regression (CWGAN-GP) (Dissertation)": lr_model_data_cwgan,
                "Random Forest Classifier (CWGAN-GP) (Dissertation)": rf_model_data_cwgan,
                "XGBoost Classifier (CWGAN-GP) (Dissertation)": xgb_model_data_cwgan,

                "Logistic Regression (SN-CWGAN-GP) (Dissertation)": lr_model_data_cwgan_reg_ya1,
                "Random Forest Classifier (SN-CWGAN-GP) (Dissertation)": rf_model_data_cwgan_reg_ya1,
                "XGBoost Classifier (SN-CWGAN-GP) (Dissertation)": xgb_model_data_cwgan_reg_ya1,
    }
    st.session_state["saved_models"] = models

def display_evaluation_matrices(cf, set="Test"):
    accuracy, precision, recall, f1_score = CommonUtility.metric_scores(cf)
    st.markdown(f"#### Evaluation Metrices ({set} set)::")
    st.write(f"**Accuracy  ::** {accuracy}")
    st.write(f"**Precision ::** {precision}")
    st.write(f"**Recal     ::** {recall}")
    st.write(f"**F1-Score  ::** {f1_score}")

def plot_binary_confusion_matrices(cf, set="Test"):
    df = pd.DataFrame(np.around(cf, 3), columns=["No Anomaly", "Anomaly"])
    df = df.assign(anomaly=["No Anomaly", "Anomaly"])
    df.set_index(["anomaly"], inplace=True, drop=True)

    st.markdown(f"#### Binary Confusion Matrix  ({set} set)::")
    fig_corr = px.imshow(
                            df,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Viridis",
                            # title="Correlation Heatmap of Numerical Columns",
                            height=1000,
    )
    st.plotly_chart(fig_corr)

def plot_confusion_matrices(labels, predictions, set="Test"):
    cf = CommonUtility.confusion_matrix(labels, predictions)

    combined = zip(st.session_state["dataframe"][st.session_state["dataframe_columns"]["anomaly"]].values, st.session_state["dataframe"][st.session_state["dataframe_columns"]["description"]].values)
    sorted_pairs = sorted(combined, key=lambda pair: pair[0])

    anomaly_ids = dict()
    for i in sorted_pairs:
        anomaly_ids[i[0]] = i[1]
    anomaly_names = [val for key, val in anomaly_ids.items()]

    cff = cf / np.sum(cf, axis=1, keepdims=True)

    df = pd.DataFrame(np.around(cff, 3), columns=anomaly_names)
    df = df.assign(anomaly=anomaly_names)
    df.set_index(["anomaly"], inplace=True, drop=True)

    st.markdown(f"#### Comprehensive Confusion Matrix ({set} set)::")
    fig_corr = px.imshow(
                            df,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Viridis",
                            # title="Correlation Heatmap of Numerical Columns",
                            height=1000,
    )
    st.plotly_chart(fig_corr)

    cff = CommonUtility.format_confusion_matrix(cf)
    cff = cff / np.sum(cff, axis=1, keepdims=True)
    plot_binary_confusion_matrices(cff)

    display_evaluation_matrices(cff)

    st.markdown("---")

def model_evaluation_page():
    st.title("Model Evaluation")
    st.write("Perform model evaluation on a dataset.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Select a CSV file from your local machine.")

    if uploaded_file is not None:
        dataframe = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

        if "datetime" in dataframe.columns:
            dataframe.set_index("datetime", inplace=True, drop=True)

        st.dataframe(dataframe)

        # model = st.session_state["saved_models"]["Logistic Regression (Dissertation)"]["model"]
        # scaler = st.session_state["saved_models"]["Logistic Regression (Dissertation)"]["scaler"]

        # values = scaler.transform(dataframe.drop(columns=["anomaly", "description"]).values)
        # dataframe_test_scaled = pd.DataFrame(values, columns=dataframe.drop(columns=["anomaly", "description"]).columns)
        # dataframe_test_scaled = dataframe_test_scaled.assign(anomaly=dataframe.anomaly.values, description=dataframe.description.values)
        # st.dataframe(dataframe_test_scaled)

        # X_test_scaled = dataframe.drop(columns=["anomaly", "description"]).values
        # y_test = dataframe["anomaly"].values.ravel()

        # y_pred = model.predict(X_test_scaled)
        # accuracy = accuracy_score(y_test, y_pred)
        # st.write(f"Accuracy: {accuracy:.4f}")

        model_list = list(st.session_state["saved_models"].keys())
        if "trained_models" in st.session_state:
            model_list = model_list + list(st.session_state["trained_models"].keys())
            
        selected_models = st.multiselect(
                                            "Please select the model(s) to evaluate ::",
                                            model_list,
                                            placeholder=None,
                                            max_selections=2,
                                            key=701
        )

        for selected_model in selected_models:
            st.markdown(f"## Model :: {selected_model}")

            if selected_model in st.session_state["saved_models"]:
                model = st.session_state["saved_models"][selected_model]["model"]
                scaler = st.session_state["saved_models"][selected_model]["scaler"]

                predictions, labels = test_model(
                                                            model, scaler, 
                                                            dataframe,
                                                            st.session_state["dataframe_columns"]["anomaly"],
                                                            st.session_state["dataframe_columns"]["description"],
                                                            use_scaler=False
                )
                plot_confusion_matrices(labels, predictions, set="Test")

            elif selected_model in st.session_state["trained_models"]:
                model = st.session_state["trained_models"][selected_model]["model"]
                scaler = st.session_state["trained_models"][selected_model]["scaler"]

                predictions, labels = test_model(
                                                            model, scaler, 
                                                            dataframe,
                                                            st.session_state["dataframe_columns"]["anomaly"],
                                                            st.session_state["dataframe_columns"]["description"],
                                                            use_scaler=False
                )
                plot_confusion_matrices(labels, predictions, set="Test")

    else:
        st.success("Please upload a dataset to evaluate.")

# Entry point for the Streamlit application
if __name__ == "__main__":
    load_models()
    model_evaluation_page()