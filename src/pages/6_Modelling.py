import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle

from datetime import datetime

from models.statistical import z_score_method, multivariate_gaussian_model
from models.statistical import interquartile_range_method, percentile_method
from models.statistical import k_nearest_neighbors, local_outlier_factor
from models.supervised import test_model
from models.supervised import logistic_regression, support_vector_machine_classifier, extreme_gradient_boosting_classifier, random_forest_classifier
from models.unsupervised import test_unsupervised_model
from models.unsupervised import isolation_forest, one_class_support_vector_machine

from utility.common_utility import CommonUtility

st.set_page_config(
    page_title="Modelling",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)

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

    combined = zip(st.session_state["dataframe_train"][st.session_state["dataframe_columns"]["anomaly"]].values, st.session_state["dataframe_train"][st.session_state["dataframe_columns"]["description"]].values)
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

def save_model(model, scaler, filename="model"):
    model_data = {
                    'model': model,
                    'scaler': scaler
    }
    with open(f'./src/models/trained/{filename}.pickle', 'wb') as handle:
        pickle.dump(model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not "trained_models" in  st.session_state: st.session_state["trained_models"] = dict()
    st.session_state["trained_models"][filename] = model_data


def modelling_page():

    st.title("Modelling")
    st.write("Perform basic modelling on the dataset.")

    # Check if a DataFrame is available in session state
    if "dataframe" in st.session_state:
        dataframe = st.session_state["dataframe"]

        tab1, tab2 = st.tabs([
                                        "Statistical", 
                                        "Machine Learning", 
                                        # "Deep Learning", 
        ])

        with tab1:
            with st.container():
                choice = st.radio(label='Choose Type of Model ::', 
                                  options=["Parametric Methods", "Non-Parametric Methods", "Proximity-Based Methods"], 
                                  horizontal=True)
                
                if choice == "Parametric Methods":
                    model_techniques = {
                                        "Z-Score Method":z_score_method,
                                        "Multivariate Gaussian Model":multivariate_gaussian_model,
                    }

                    selected_pm_method = st.selectbox(
                                                        "Select a modelling technique ::", 
                                                        model_techniques.keys(), 
                                                        key=601
                    )

                    if selected_pm_method == "Z-Score Method":

                        selected_threshold = st.slider(
                            "Z-score Threshold for Anomaly Detection:",
                            min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                            help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )

                        selected_set = st.selectbox(
                                                        "Select evaluation dataset ::", 
                                                        ["Test", "Validation", "Train"], 
                                                        key=602
                        )

                        dataframe_eval = st.session_state["dataframe_test"]
                        if selected_set == "Validation": dataframe_eval = st.session_state["dataframe_val"]
                        elif selected_set == "Train": dataframe_eval = st.session_state["dataframe_train"]

                        predictions, labels = model_techniques[selected_pm_method](
                                                                                st.session_state["dataframe_train"],
                                                                                st.session_state["dataframe_columns"]["anomaly"],
                                                                                st.session_state["dataframe_columns"]["description"],
                                                                                selected_threshold,
                                                                                dataframe_eval,
                        )
                        
                        cf = CommonUtility.confusion_matrix(labels, predictions)
                        cff = cf / np.sum(cf, axis=1, keepdims=True)
                        plot_binary_confusion_matrices(cff, set=selected_set)
                        display_evaluation_matrices(cff, set=selected_set)

                        st.markdown("---")

                    if selected_pm_method == "Multivariate Gaussian Model":
                        selected_contamination = st.slider(
                            "Set Contamination:",
                            min_value=0.01, max_value=1.00, value=0.3, step=0.01,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        robust_covariance = st.checkbox("Robust Covariance", value=False)

                        selected_set = st.selectbox(
                                                        "Select evaluation dataset ::", 
                                                        ["Test", "Validation", "Train"], 
                                                        key=603
                        )

                        dataframe_eval = st.session_state["dataframe_test"]
                        if selected_set == "Validation": dataframe_eval = st.session_state["dataframe_val"]
                        elif selected_set == "Train": dataframe_eval = st.session_state["dataframe_train"]

                        predictions, labels = model_techniques[selected_pm_method](
                                                                                st.session_state["dataframe_train"],
                                                                                st.session_state["dataframe_columns"]["anomaly"],
                                                                                st.session_state["dataframe_columns"]["description"],
                                                                                selected_contamination,
                                                                                robust_covariance,
                                                                                dataframe_eval,
                        )
                        
                        cf = CommonUtility.confusion_matrix(labels, predictions)
                        cff = cf / np.sum(cf, axis=1, keepdims=True)
                        plot_binary_confusion_matrices(cff, set=selected_set)
                        display_evaluation_matrices(cff, set=selected_set)

                        st.markdown("---")
                
                if choice == "Non-Parametric Methods":
                    model_techniques = {
                                        "Interquartile Range (IQR) Method":interquartile_range_method,
                                        "Percentile Method":percentile_method,
                    }

                    selected_npm_method = st.selectbox(
                                                        "Select a modelling technique ::", 
                                                        model_techniques.keys(), 
                                                        key=605
                    )

                    if selected_npm_method == "Interquartile Range (IQR) Method":
                        selected_iqr_multiplier = st.slider(
                            "Set IQR Multiplier:",
                            min_value=0.5, max_value=3.00, value=1.5, step=0.1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )

                        selected_set = st.selectbox(
                                                        "Select evaluation dataset ::", 
                                                        ["Test", "Validation", "Train"], 
                                                        key=606
                        )

                        dataframe_eval = st.session_state["dataframe_test"]
                        if selected_set == "Validation": dataframe_eval = st.session_state["dataframe_val"]
                        elif selected_set == "Train": dataframe_eval = st.session_state["dataframe_train"]

                        predictions, labels = model_techniques[selected_npm_method](
                                                                                st.session_state["dataframe_train"],
                                                                                st.session_state["dataframe_columns"]["anomaly"],
                                                                                st.session_state["dataframe_columns"]["description"],
                                                                                selected_iqr_multiplier,
                                                                                dataframe_eval,
                        )
                        
                        cf = CommonUtility.confusion_matrix(labels, predictions)
                        cff = cf / np.sum(cf, axis=1, keepdims=True)
                        plot_binary_confusion_matrices(cff, set=selected_set)
                        display_evaluation_matrices(cff, set=selected_set)

                        st.markdown("---")

                    if selected_npm_method == "Percentile Method":
                        selected_percentile_range = st.slider(
                            "Select a percentile range", 
                            0.0, 100.0, (1.0, 99.0)
                        )

                        selected_set = st.selectbox(
                                                        "Select evaluation dataset ::", 
                                                        ["Test", "Validation", "Train"], 
                                                        key=607
                        )

                        dataframe_eval = st.session_state["dataframe_test"]
                        if selected_set == "Validation": dataframe_eval = st.session_state["dataframe_val"]
                        elif selected_set == "Train": dataframe_eval = st.session_state["dataframe_train"]

                        predictions, labels = model_techniques[selected_npm_method](
                                                                                st.session_state["dataframe_train"],
                                                                                st.session_state["dataframe_columns"]["anomaly"],
                                                                                st.session_state["dataframe_columns"]["description"],
                                                                                selected_percentile_range[0],
                                                                                selected_percentile_range[1],
                                                                                dataframe_eval,
                        )
                        
                        cf = CommonUtility.confusion_matrix(labels, predictions)
                        cff = cf / np.sum(cf, axis=1, keepdims=True)
                        plot_binary_confusion_matrices(cff, set=selected_set)
                        display_evaluation_matrices(cff, set=selected_set)

                        st.markdown("---")
                
                if choice == "Proximity-Based Methods":
                    model_techniques = {
                                        "k-Nearest Neighbors":k_nearest_neighbors,
                                        "Local Outlier Factor":local_outlier_factor,
                    }

                    selected_pb_method = st.selectbox(
                                                        "Select a modelling technique ::", 
                                                        model_techniques.keys(), 
                                                        key=608
                    )

                    if selected_pb_method == "k-Nearest Neighbors":
                        selected_n_neighbors = st.slider(
                            "N-Neighbors for kNN:",
                            min_value=1, max_value=100, value=10, step=1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )

                        selected_contamination = st.slider(
                            "Set Contamination:",
                            min_value=0.01, max_value=1.00, value=0.3, step=0.01,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )

                        selected_metric = st.selectbox(
                                                        "Select distance metric ::", 
                                                        ['euclidean', 'manhattan', 'minkowski'], 
                                                        key=609
                        )

                        selected_set = st.selectbox(
                                                        "Select evaluation dataset ::", 
                                                        ["Test", "Validation", "Train"], 
                                                        key=610
                        )

                        dataframe_eval = st.session_state["dataframe_test"]
                        if selected_set == "Validation": dataframe_eval = st.session_state["dataframe_val"]
                        elif selected_set == "Train": dataframe_eval = st.session_state["dataframe_train"]

                        predictions, labels = model_techniques[selected_pb_method](
                                                                                st.session_state["dataframe_train"],
                                                                                st.session_state["dataframe_columns"]["anomaly"],
                                                                                st.session_state["dataframe_columns"]["description"],
                                                                                selected_n_neighbors,
                                                                                selected_contamination,
                                                                                selected_metric,
                                                                                dataframe_eval,
                        )
                        
                        cf = CommonUtility.confusion_matrix(labels, predictions)
                        cff = cf / np.sum(cf, axis=1, keepdims=True)
                        plot_binary_confusion_matrices(cff, set=selected_set)
                        display_evaluation_matrices(cff, set=selected_set)

                        st.markdown("---")

                    if selected_pb_method == "Local Outlier Factor":
                        selected_n_neighbors = st.slider(
                            "N-Neighbors for kNN:",
                            min_value=1, max_value=100, value=10, step=1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )

                        selected_contamination = st.slider(
                            "Set Contamination:",
                            min_value=0.01, max_value=0.50, value=0.3, step=0.01,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )

                        selected_set = st.selectbox(
                                                        "Select evaluation dataset ::", 
                                                        ["Test", "Validation", "Train"], 
                                                        key=611
                        )

                        dataframe_eval = st.session_state["dataframe_test"]
                        if selected_set == "Validation": dataframe_eval = st.session_state["dataframe_val"]
                        elif selected_set == "Train": dataframe_eval = st.session_state["dataframe_train"]

                        predictions, labels = model_techniques[selected_pb_method](
                                                                                st.session_state["dataframe_train"],
                                                                                st.session_state["dataframe_columns"]["anomaly"],
                                                                                st.session_state["dataframe_columns"]["description"],
                                                                                selected_n_neighbors,
                                                                                selected_contamination,
                                                                                dataframe_eval,
                        )
                        
                        cf = CommonUtility.confusion_matrix(labels, predictions)
                        cff = cf / np.sum(cf, axis=1, keepdims=True)
                        plot_binary_confusion_matrices(cff, set=selected_set)
                        display_evaluation_matrices(cff, set=selected_set)

                        st.markdown("---")

        with tab2:
            with st.container():
                choice = st.radio(label='Choose Type of Model ::', options=["Supervised", "Unsupervised"], horizontal=True)

                if choice == 'Supervised':
                    model_techniques = {
                                        "Logistic Regression":logistic_regression,
                                        "Support Vector Machine Classifer":support_vector_machine_classifier,
                                        "Random Forest Classifier":random_forest_classifier,
                                        "Extreme Gradient Boosting Classifier":extreme_gradient_boosting_classifier,
                    }

                    selected_supervisied_method = st.selectbox(
                                                                "Select a modelling technique ::", 
                                                                model_techniques.keys(), 
                                                                key=612
                    )

                    if selected_supervisied_method == "Logistic Regression":
                        st.markdown("#### Model Parameters ::")
                        selected_solver = st.selectbox(
                                                        "Select a Logistic Regression solver ::", 
                                                        ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 
                                                        key=613
                        )

                        selected_C = st.slider(
                            "Set C Value:",
                            min_value=1, max_value=20, value=1, step=1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        model, scaler = model_techniques[selected_supervisied_method](
                                                                                        st.session_state["dataframe_train"],
                                                                                        st.session_state["dataframe_columns"]["anomaly"],
                                                                                        st.session_state["dataframe_columns"]["description"],
                                                                                        selected_solver,
                                                                                        selected_C,
                                                                                        use_scaler=False
                        )

                        now = datetime.now()
                        filename = f"logistic_regression_{selected_solver}_{selected_C}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
                        is_save_download = st.button(
                                                        "Save Model", 
                                                        key=617, 
                                                        help="Click to save the trained model as a pickle file and download it.",
                                                        type="primary",
                                                        on_click=save_model, args=(model, scaler, filename))
                        if is_save_download:
                            st.download_button(
                                label="Download Model",
                                data=pickle.dumps({'model': model, 'scaler': scaler}),
                                file_name=f"{filename}.pickle",
                                mime="application/octet-stream" # Optional: specify the MIME type
                            )

                    elif selected_supervisied_method == "Support Vector Machine Classifer":
                        st.markdown("#### Model Parameters ::")
                        selected_kernel = st.selectbox(
                                                        "Select a Support Vector Machine Classifer kernel ::", 
                                                        ['linear', 'poly', 'rbf', 'sigmoid'], 
                                                        key=614
                        )

                        selected_C = st.slider(
                            "Set C Value:",
                            min_value=1, max_value=20, value=1, step=1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        model, scaler = model_techniques[selected_supervisied_method](
                                                                                        st.session_state["dataframe_train"],
                                                                                        st.session_state["dataframe_columns"]["anomaly"],
                                                                                        st.session_state["dataframe_columns"]["description"],
                                                                                        selected_kernel,
                                                                                        selected_C,
                                                                                        use_scaler=False
                        )

                        now = datetime.now()
                        filename = f"svm_classifier_{selected_kernel}_{selected_C}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
                        is_save_download = st.button(
                                                        "Save Model", 
                                                        key=617, 
                                                        help="Click to save the trained model as a pickle file and download it.",
                                                        type="primary",
                                                        on_click=save_model, args=(model, scaler, filename))
                        if is_save_download:
                            st.download_button(
                                label="Download Model",
                                data=pickle.dumps({'model': model, 'scaler': scaler}),
                                file_name=f"{filename}.pickle",
                                mime="application/octet-stream" # Optional: specify the MIME type
                            )

                    elif selected_supervisied_method == "Random Forest Classifier":
                        st.markdown("#### Model Parameters ::")
                        selected_criterion = st.selectbox(
                                                        "Select a Support Vector Machine Classifer kernel ::", 
                                                        ['gini', 'entropy', 'log_loss'], 
                                                        key=615
                        )

                        selected_n_estimator = st.slider(
                            "Set N Estimators:",
                            min_value=10, max_value=100, value=10, step=1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        model, scaler = model_techniques[selected_supervisied_method](
                                                                                        st.session_state["dataframe_train"],
                                                                                        st.session_state["dataframe_columns"]["anomaly"],
                                                                                        st.session_state["dataframe_columns"]["description"],
                                                                                        selected_criterion,
                                                                                        selected_n_estimator,
                                                                                        use_scaler=False
                        )

                        now = datetime.now()
                        filename = f"rf_classifier_{selected_criterion}_{selected_n_estimator}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
                        is_save_download = st.button(
                                                        "Save Model", 
                                                        key=617, 
                                                        help="Click to save the trained model as a pickle file and download it.",
                                                        type="primary",
                                                        on_click=save_model, args=(model, scaler, filename))
                        if is_save_download:
                            st.download_button(
                                label="Download Model",
                                data=pickle.dumps({'model': model, 'scaler': scaler}),
                                file_name=f"{filename}.pickle",
                                mime="application/octet-stream" # Optional: specify the MIME type
                            )

                    elif selected_supervisied_method == "Extreme Gradient Boosting Classifier":
                        st.markdown("#### Model Parameters ::")
                        selected_lr = st.slider(
                            "Set Learning Rate:",
                            min_value=0.01, max_value=0.3, value=0.01, step=0.01,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        selected_n_estimator = st.slider(
                            "Set N Estimators:",
                            min_value=10, max_value=500, value=10, step=1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        model, scaler = model_techniques[selected_supervisied_method](
                                                                                        st.session_state["dataframe_train"],
                                                                                        st.session_state["dataframe_columns"]["anomaly"],
                                                                                        st.session_state["dataframe_columns"]["description"],
                                                                                        selected_lr,
                                                                                        selected_n_estimator,
                                                                                        use_scaler=False
                        )

                        now = datetime.now()
                        filename = f"xgb_classifier_{selected_n_estimator}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
                        is_save_download = st.button(
                                                        "Save Model", 
                                                        key=617, 
                                                        help="Click to save the trained model as a pickle file and download it.",
                                                        type="primary",
                                                        on_click=save_model, args=(model, scaler, filename))
                        if is_save_download:
                            st.download_button(
                                label="Download Model",
                                data=pickle.dumps({'model': model, 'scaler': scaler}),
                                file_name=f"{filename}.pickle",
                                mime="application/octet-stream" # Optional: specify the MIME type
                            )

                    selected_set = st.selectbox(
                                                    "Select evaluation dataset ::", 
                                                    ["Test", "Validation", "Train"], 
                                                    key=616
                    )

                    if selected_set == "Train":
                        predictions, labels = test_model(
                                                                    model, scaler, 
                                                                    st.session_state["dataframe_train"],
                                                                    st.session_state["dataframe_columns"]["anomaly"],
                                                                    st.session_state["dataframe_columns"]["description"],
                                                                    use_scaler=False
                        )
                    elif selected_set == "Validation":
                        predictions, labels = test_model(
                                                                    model, scaler, 
                                                                    st.session_state["dataframe_val"],
                                                                    st.session_state["dataframe_columns"]["anomaly"],
                                                                    st.session_state["dataframe_columns"]["description"],
                                                                    use_scaler=False
                        )
                    elif selected_set == "Test":
                        predictions, labels = test_model(
                                                                    model, scaler, 
                                                                    st.session_state["dataframe_test"],
                                                                    st.session_state["dataframe_columns"]["anomaly"],
                                                                    st.session_state["dataframe_columns"]["description"],
                                                                    use_scaler=False
                        )
                    plot_confusion_matrices(labels, predictions, set=selected_set)

                if choice == 'Unsupervised':
                    model_techniques = {
                                        "Isolation Forest":isolation_forest,
                                        "One Class Support Vector Machine":one_class_support_vector_machine,
                    }

                    selected_unsupervisied_method = st.selectbox(
                                                                    "Select a modelling technique ::", 
                                                                    model_techniques.keys(), 
                                                                    key=617
                    )

                    if selected_unsupervisied_method == "Isolation Forest":
                        st.markdown("#### Model Parameters ::")
                        selected_contamination = st.slider(
                            "Set Contamination:",
                            min_value=0.01, max_value=1.00, value=0.3, step=0.01, key=618
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )

                        selected_n_estimator = st.slider(
                            "Set N Estimators:",
                            min_value=10, max_value=100, value=10, step=1,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        model, scaler = model_techniques[selected_unsupervisied_method](
                                                                                        st.session_state["dataframe_train"],
                                                                                        st.session_state["dataframe_columns"]["anomaly"],
                                                                                        st.session_state["dataframe_columns"]["description"],
                                                                                        selected_contamination,
                                                                                        selected_n_estimator
                        )

                    elif selected_unsupervisied_method == "One Class Support Vector Machine":
                        st.markdown("#### Model Parameters ::")
                        selected_kernel = st.selectbox(
                                                        "Select a One Class Support Vector Machine Classifer kernel ::", 
                                                        ['linear', 'poly', 'rbf', 'sigmoid'], 
                                                        key=619
                        )

                        selected_nu = st.slider(
                            "Set nu Value:",
                            min_value=0.01, max_value=1.00, value=0.01, step=0.01,
                            # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                        )
                        model, scaler = model_techniques[selected_unsupervisied_method](
                                                                                        st.session_state["dataframe_train"],
                                                                                        st.session_state["dataframe_columns"]["anomaly"],
                                                                                        st.session_state["dataframe_columns"]["description"],
                                                                                        selected_kernel,
                                                                                        selected_nu
                        )

                    selected_set = st.selectbox(
                                                    "Select evaluation dataset ::", 
                                                    ["Test", "Validation", "Train"], 
                                                    key=620
                    )

                    if selected_set == "Train":
                        predictions, labels = test_unsupervised_model(
                                                                    model, scaler, 
                                                                    st.session_state["dataframe_train"],
                                                                    st.session_state["dataframe_columns"]["anomaly"],
                                                                    st.session_state["dataframe_columns"]["description"],
                        )
                    elif selected_set == "Validation":
                        predictions, labels = test_unsupervised_model(
                                                                    model, scaler, 
                                                                    st.session_state["dataframe_val"],
                                                                    st.session_state["dataframe_columns"]["anomaly"],
                                                                    st.session_state["dataframe_columns"]["description"],
                        )
                    elif selected_set == "Test":
                        predictions, labels = test_unsupervised_model(
                                                                    model, scaler, 
                                                                    st.session_state["dataframe_test"],
                                                                    st.session_state["dataframe_columns"]["anomaly"],
                                                                    st.session_state["dataframe_columns"]["description"],
                        )
                        
                    cf = CommonUtility.confusion_matrix(labels, predictions)
                    cff = cf / np.sum(cf, axis=1, keepdims=True)
                    plot_binary_confusion_matrices(cff, set=selected_set)
                    display_evaluation_matrices(cff, set=selected_set)

                st.markdown("---")

    else:
        st.warning("No data uploaded yet! Please go back to the 'Upload Data' page to upload a CSV file.")
        st.markdown("---")
        st.markdown("### How to Use This Page:")
        st.markdown("1. Navigate to the 'Data Uploader' page (usually the first page in the sidebar).")
        st.markdown("2. Upload your CSV file there.")
        st.markdown("3. Once uploaded, return to this 'Exploratory Data Analysis' page to see the analysis.")

# Entry point for this specific page
if __name__ == "__main__":
    modelling_page()