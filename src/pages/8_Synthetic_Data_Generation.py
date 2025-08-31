import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE, ADASYN

from page_data.synthetic_data_generation import CWGAN_GP_Generator, SN_CWGAN_GP_Generator

st.set_page_config(
    page_title="Synthetic Data Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)
            
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

def generate_samples(generator, label, num_samples, batch_size, latent_dimension, device):
    generator.eval()

    generated_list = list()
    for _ in range(num_samples // batch_size):
        auto_label = torch.full((batch_size, 1), label, dtype=torch.int64, device=device)
        noise = torch.randn(batch_size, latent_dimension).to(device)
        with torch.no_grad():
            generated_samples = generator(noise, auto_label).cpu()
            generated_list.append(generated_samples)

    return np.vstack(generated_list)

def display_stats(df):
    with st.container():
        st.subheader("Dataframe ::")
        st.dataframe(df)

        st.markdown("---")

        st.subheader("Basic Statistics ::")

        # Display descriptive statistics of the numerical columns
        st.write(df.describe())

        st.markdown("---")

        st.subheader("Data Information ::")

        # Display information about columns, including data types and non-null values
        # Using st.write(dataframe.info()) directly doesn't work well for display;
        # A better way is to iterate or convert info to a string or table.
        # For simplicity, let's show dtypes and shape.
        st.write(f"**Number of rows:** {df.shape[0]}")
        st.write(f"**Number of columns:** {df.shape[1]}")

        st.write("**Column Data Types:**")
        st.write(df.dtypes)

        st.markdown("---")

def sdg_page():

    st.title("Synthetic Data Generation")
    st.write("Generate synthetic data for development purposes.")

    # Check if a DataFrame is available in session state
    if "dataframe" in st.session_state:
        dataframe = st.session_state["dataframe"]

        # fig = plt.figure()
        # dataframe[st.session_state["dataframe_columns"]["description"]].value_counts().plot(kind='bar')
        # st.pyplot()

        # fig = plt.figure()
        # dataframe[st.session_state["dataframe_columns"]["description"]].value_counts().plot(kind='bar')
        # st.pyplot()

        value_count_df = dataframe[st.session_state["dataframe_columns"]["description"]].value_counts()
        # st.dataframe(value_count_df)

        data = {
            'Description': value_count_df.index,  # Initializes all checkboxes to False
            'Count': value_count_df.values,
            'Synthetic Data To Be Generated': np.abs(value_count_df.values - np.max(value_count_df.values))
        }
        df = pd.DataFrame(data)
        st.data_editor(
            df,
            hide_index=True,
            column_config={
                "Synthetic Data To Be Generated": st.column_config.NumberColumn(
                    "Synthetic Data To Be Generated",
                    help="Number of synthetic data points to be generated",
                    min_value=0,
                    step=1,
                    format="%d",
                    width='small',
                ),
            },
            disabled=("Description", "Count") # Disable editing for these columns
        )

        selected_method = st.selectbox(
                                "Select evaluation dataset ::", 
                                [
                                    "Synthetic Minority Over-sampling Technique (SMOTE)", 
                                    "Adaptive Synthetic Sampling (ADASYN)", 
                                    "Conditional Wasserstein Generative Adversarial Networks with Gradient Penalty (CWGAN-GP)",
                                    "Spectral Normalized Conditional Wasserstein Generative Adversarial Networks with Gradient Penalty (SN-CWGAN-GP)"
                                ], 
                                key=801,
        )
        if selected_method == "Synthetic Minority Over-sampling Technique (SMOTE)":
            smote = SMOTE(random_state=42)
            X_res_smote, y_res_smote = smote.fit_resample(dataframe.drop(columns=["anomaly", "description"]), dataframe.anomaly.values)

            dataframe_smote = pd.DataFrame(X_res_smote, columns=dataframe.drop(columns=["anomaly", "description"]).columns)
            dataframe_smote["anomaly"] = y_res_smote

            anomaly_ids = dict()
            for idx, desc in zip(dataframe.anomaly, dataframe.description):
                if not idx in anomaly_ids:
                    anomaly_ids[idx] = desc
            
            desc = list()
            for idx in dataframe_smote.anomaly:
                desc.append(anomaly_ids[idx])
            dataframe_smote["description"] = desc

            display_stats(dataframe_smote)

            st.download_button(
                                    label="Download",
                                    data=convert_df_to_csv(dataframe_smote),
                                    file_name="SMOTE_Augmented_Dataset.csv",  # This sets the downloaded file's name
                                    mime="text/csv",
            )

        elif selected_method == "Adaptive Synthetic Sampling (ADASYN)":
            adasyn = ADASYN(random_state=42)
            X_res_adasyn, y_res_adasyn = adasyn.fit_resample(dataframe.drop(columns=["anomaly", "description"]), dataframe.anomaly.values)

            dataframe_adasyn = pd.DataFrame(X_res_adasyn, columns=dataframe.drop(columns=["anomaly", "description"]).columns)
            dataframe_adasyn["anomaly"] = y_res_adasyn

            anomaly_ids = dict()
            for idx, desc in zip(dataframe.anomaly, dataframe.description):
                if not idx in anomaly_ids:
                    anomaly_ids[idx] = desc
            
            desc = list()
            for idx in dataframe_adasyn.anomaly:
                desc.append(anomaly_ids[idx])
            dataframe_adasyn["description"] = desc

            display_stats(dataframe_adasyn)

            st.download_button(
                                    label="Download",
                                    data=convert_df_to_csv(dataframe_adasyn),
                                    file_name="ADASYN_Augmented_Dataset.csv",  # This sets the downloaded file's name
                                    mime="text/csv",
            )
        
        elif selected_method == "Conditional Wasserstein Generative Adversarial Networks with Gradient Penalty (CWGAN-GP)":
            LATENT_DIMENSION, FEATURE_SIZE, BATCH_SIZE = 16, 8, 512
            DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            generator = CWGAN_GP_Generator(LATENT_DIMENSION, FEATURE_SIZE).to(DEVICE)
            generator_path = './src/models/saved/generator_cwgan_510.pth'
            generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))
            generator.eval()

            anomaly_ids = dict()
            for idx, desc in zip(dataframe.anomaly, dataframe.description):
                if not idx in anomaly_ids:
                    anomaly_ids[idx] = desc

            samples, anomaly, description = list(), list(), list()
            for idx in dataframe.anomaly.unique():
                if idx == 0.0: continue

                num_samples_generate = len(dataframe.iloc[np.where(dataframe.anomaly==0.0)]) - len(dataframe.iloc[np.where(dataframe.anomaly==idx)])
                generated_samples = generate_samples(generator, idx, num_samples_generate, BATCH_SIZE, LATENT_DIMENSION, DEVICE)

                samples.append(generated_samples)
                anomaly = anomaly + [idx]*generated_samples.shape[0]
                description = description + [anomaly_ids[idx]]*generated_samples.shape[0]

            tmp = dataframe.drop(columns=["anomaly", "description"], axis=1)
            scaler = StandardScaler()
            scaler.fit_transform(tmp)

            generated_samples = pd.DataFrame(scaler.inverse_transform(np.vstack(samples)), columns=st.session_state["dataframe_columns"]["numerical"])
            generated_samples["description"] = description
            generated_samples["anomaly"] = anomaly

            augmented_samples = pd.concat([dataframe.reset_index(drop=True), generated_samples], axis=0)
            display_stats(augmented_samples)

            st.download_button(
                                    label="Download",
                                    data=convert_df_to_csv(augmented_samples),
                                    file_name="GAN_GP_Augmented_Dataset.csv",  # This sets the downloaded file's name
                                    mime="text/csv",
            )
        
        elif selected_method == "Spectral Normalized Conditional Wasserstein Generative Adversarial Networks with Gradient Penalty (SN-CWGAN-GP)":
            LATENT_DIMENSION, FEATURE_SIZE, BATCH_SIZE = 64, 8, 512
            DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            generator = SN_CWGAN_GP_Generator(LATENT_DIMENSION, FEATURE_SIZE).to(DEVICE)
            generator_path = './src/models/saved/generator_cwgan_reg_ya1_100.pth'
            generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))
            generator.eval()

            anomaly_ids = dict()
            for idx, desc in zip(dataframe.anomaly, dataframe.description):
                if not idx in anomaly_ids:
                    anomaly_ids[idx] = desc

            samples, anomaly, description = list(), list(), list()
            for idx in dataframe.anomaly.unique():
                if idx == 0.0: continue

                num_samples_generate = len(dataframe.iloc[np.where(dataframe.anomaly==0.0)]) - len(dataframe.iloc[np.where(dataframe.anomaly==idx)])
                generated_samples = generate_samples(generator, idx, num_samples_generate, BATCH_SIZE, LATENT_DIMENSION, DEVICE)

                samples.append(generated_samples)
                anomaly = anomaly + [idx]*generated_samples.shape[0]
                description = description + [anomaly_ids[idx]]*generated_samples.shape[0]

            tmp = dataframe.drop(columns=["anomaly", "description"], axis=1)
            scaler = StandardScaler()
            scaler.fit_transform(tmp)

            generated_samples = pd.DataFrame(scaler.inverse_transform(np.vstack(samples)), columns=st.session_state["dataframe_columns"]["numerical"])
            generated_samples["description"] = description
            generated_samples["anomaly"] = anomaly

            augmented_samples = pd.concat([dataframe.reset_index(drop=True), generated_samples], axis=0)
            display_stats(augmented_samples)

            st.download_button(
                                    label="Download",
                                    data=convert_df_to_csv(augmented_samples),
                                    file_name="SN_GAN_GP_Augmented_Dataset.csv",  # This sets the downloaded file's name
                                    mime="text/csv",
            )

# Entry point for the Streamlit application
if __name__ == "__main__":
    sdg_page()