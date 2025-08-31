import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import pickle

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

from models.supervised import test_model

import lime
import lime.lime_tabular

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)

def load_data_and_model_for_lime(dataframe):
    """Generates synthetic data, trains a Random Forest model, and creates a LIME explainer."""
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=dataframe.drop(columns=[st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]]).values,
        feature_names=dataframe.drop(columns=[st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]]).columns.tolist(),
        class_names=dataframe[st.session_state["dataframe_columns"]["description"]].unique().tolist(),
        mode='classification'
    )
    return explainer

def plot_prediction(df):
    fig_bar = px.bar(
                        df, 
                        x=df['Class'], 
                        y=df['Probability'], 
                        hover_data=["Probability"],
                        color=df['Class'], 
                        title=f"Value Counts of {st.session_state['dataframe_columns']['description']}",
                        height=600,
    )
    st.plotly_chart(fig_bar)

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

def anomaly_detection_page():
    st.title("Anomaly Detection Analysis")
    st.write("Anomaly detection using various Machine Learning algorithms")

    # Check if a DataFrame is available in session state
    if "dataframe" in st.session_state:
        dataframe = st.session_state["dataframe"]

        with st.container():

            model_list = list(st.session_state["saved_models"].keys())
            if "trained_models" in st.session_state:
                model_list = model_list + list(st.session_state["trained_models"].keys())
                
            selected_model = st.selectbox(
                                                "Please select the model to evaluate ::",
                                                model_list,
                                                key=701
            )

            selected_num_col_scatter_ts = st.selectbox(
                                                        "Select a categorical column for time series:", 
                                                        st.session_state["dataframe_columns"]["numerical"], 
                                                        key=902
            )
        
        with st.container():
            st.subheader("Scatter Plot with Anomalies Highlighted")
            dataframe_anomaly = dataframe.iloc[np.where(dataframe[st.session_state["dataframe_columns"]["anomaly"]]!=0)]

            fig = px.scatter(
                                dataframe_anomaly, 
                                x=dataframe_anomaly.index, 
                                y=selected_num_col_scatter_ts, 
                                color=st.session_state["dataframe_columns"]["description"],
                                symbol=st.session_state["dataframe_columns"]["description"],
            )

            fig.add_trace(
                            go.Scatter(
                                            y=dataframe[selected_num_col_scatter_ts],
                                            x=dataframe.index,
                                            # mode='lines',
                                            name=selected_num_col_scatter_ts,
                                            line=dict(color='blue')
            ))

            st.plotly_chart(fig, key=903)
        
        with st.container():
            st.subheader("Scatter Plot with Predicted Anomalies Highlighted")

            if selected_model in st.session_state["saved_models"]:
                model = st.session_state["saved_models"][selected_model]["model"]
                scaler = st.session_state["saved_models"][selected_model]["scaler"]

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

            anomaly_ids = dict()
            for idx, desc in zip(dataframe.anomaly, dataframe.description):
                if not idx in anomaly_ids:
                    anomaly_ids[idx] = desc
                    
            dataframe_copy = dataframe.drop(columns=[st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]]).copy()
            dataframe_copy[st.session_state["dataframe_columns"]["anomaly"]] = predictions
            dataframe_copy[st.session_state["dataframe_columns"]["description"]] = dataframe_copy[st.session_state["dataframe_columns"]["anomaly"]].map(anomaly_ids)

            dataframe_anomaly_copy = dataframe_copy.iloc[np.where(dataframe_copy[st.session_state["dataframe_columns"]["anomaly"]]!=0)]

            fig = px.scatter(
                                dataframe_anomaly_copy, 
                                x=dataframe_anomaly_copy.index, 
                                y=selected_num_col_scatter_ts, 
                                color=st.session_state["dataframe_columns"]["description"],
                                symbol=st.session_state["dataframe_columns"]["description"],
            )

            fig.add_trace(
                            go.Scatter(
                                            y=dataframe_copy[selected_num_col_scatter_ts],
                                            x=dataframe_copy.index,
                                            # mode='lines',
                                            name=selected_num_col_scatter_ts,
                                            line=dict(color='blue')
            ))

            st.plotly_chart(fig, key=904)

        # with st.container():
        #     st.subheader("Local Interpretable Model-agnostic Explanations (LIME) for Anomaly Detection")

        #     explainer_lime = load_data_and_model_for_lime(dataframe)
        #     dataframe_test = dataframe.drop(columns=[st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]]).copy()
        #     dataframe_train = dataframe.drop(columns=[st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]]).copy()

        #     # Store LIME explanations
        #     lime_explanations = list()
        #     for idx in range(len(dataframe)):
        #         instance_to_explain = dataframe_test.iloc[idx]
        #         explanation = explainer_lime.explain_instance(
        #             data_row=instance_to_explain,
        #             predict_fn=model.predict_proba,
        #             num_features=len(dataframe_train.columns.tolist())
        #         )
        #         lime_explanations.append(explanation)

        #     # Aggregate feature importance from all explanations
        #     global_feature_importance = defaultdict(float)

        #     for exp in lime_explanations:
        #         for feature, importance in exp.as_list():
        #             global_feature_importance[feature] += importance

        #     # Sort the features by importance
        #     sorted_features = sorted(global_feature_importance.items(), key=lambda x: x[1], reverse=True)

        #     # Display the top global features
        #     st.write("Global feature importance from LIME explanations:")

        #     # Sort the features by the absolute value of importance
        #     sorted_features = sorted(global_feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

        #     # Extract the top 10 features and their importance values
        #     top_10_features = sorted_features[:10]
        #     features = [feature for feature, importance in top_10_features]
        #     importance_values = [importance for feature, importance in top_10_features]

        #     # Plot the bar chart
        #     plt.figure(figsize=(10, 6))
        #     plt.barh(features, importance_values, color='skyblue')
        #     plt.xlabel('Importance')
        #     plt.ylabel('Feature')
        #     plt.title('Top 10 Global Feature Importance from LIME Explanations')
        #     plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
        #     plt.show()

        #     st.pyplot(plt)

        with st.container():
            st.subheader("Local Interpretable Model-agnostic Explanations (LIME) for a data point")

            explainer_lime = load_data_and_model_for_lime(dataframe)

            dataframe_test = dataframe.drop(columns=[st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]]).copy()
            dataframe_train = dataframe.drop(columns=[st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]]).copy()
            
            st.dataframe(pd.DataFrame({
                                "index" : [_ for _ in range(len(dataframe_train.index))],
                                "datetime" : dataframe_train.index
            }, columns=["index", "datetime"]).set_index("index"))

            i = st.number_input("Enter an integer:", min_value=0, max_value=len(dataframe_train)-1, step=1, key=905)
            instance_to_explain = dataframe_test.iloc[i]

            # Generate an explanation for the prediction of the selected instance.
            # The `model.predict_proba` function is used by LIME to get prediction probabilities.
            explanation = explainer_lime.explain_instance(
                data_row=instance_to_explain,
                predict_fn=model.predict_proba,
                num_features=len(dataframe_train.columns.tolist())
            )

            # Visualize the explanation. This will show a table and a plot of feature contributions.
            st.write(f"Explaining instance {i} using LIME:")
            st.write(f"Actual class: {dataframe[st.session_state['dataframe_columns']['description']].iloc[i]}")
            st.write(f"Predicted class: {anomaly_ids[model.predict(instance_to_explain.values.reshape(1, -1))[0]]}")

            fig = explanation.as_pyplot_figure()
            st.pyplot(fig)

            data = {
                'Class': dataframe[st.session_state["dataframe_columns"]["description"]].unique().tolist(),
                'Probability': model.predict_proba(instance_to_explain.values.reshape(1, -1))[0],
            }
            df = pd.DataFrame(data)
            plot_prediction(df)

    else:
        st.success("Please upload a dataset in the 'Upload Data' page.")

# Entry point for the Streamlit application
if __name__ == "__main__":
    load_models()
    anomaly_detection_page()