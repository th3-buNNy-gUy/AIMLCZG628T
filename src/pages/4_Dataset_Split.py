import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from utility.common_utility import CommonUtility

st.set_page_config(
    page_title="Dataset Split",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)

def plot_data_proportions(dataframe):      
    value_counts = dataframe[st.session_state['dataframe_columns']['description']].value_counts().reset_index()
    value_counts.columns = [st.session_state['dataframe_columns']['description'], "Count"]
    value_counts["Percentage"] = value_counts["Count"] * 100 / sum(value_counts["Count"])

    fig_bar = px.bar(
                        value_counts, 
                        x=st.session_state['dataframe_columns']['description'], 
                        y="Percentage", 
                        hover_data=["Count"],
                        color=st.session_state["dataframe_columns"]["description"], 
                        title=f"Value Counts of {st.session_state['dataframe_columns']['description']}",
                        height=600,
    )
    st.plotly_chart(fig_bar)

def split():

    st.title("Dataset Split")
    st.write("Split Dataset into Train, Test & Validation sets")

    # Check if a DataFrame is available in session state
    if "dataframe" in st.session_state:
        dataframe = st.session_state["dataframe"]
    
        with st.container():
            st.markdown("### Dataset")
            st.dataframe(st.session_state["dataframe"])

            num_rows, num_cols = st.session_state["dataframe"].shape
            st.write(f"**Number of rows:** {num_rows}")
            st.write(f"**Number of columns:** {num_cols}")
            st.markdown("---")
        
        with st.container():
            st.markdown("### Split")

            selected_test_split = st.slider(
                "Percentage of the dataset to be used as Test set:",
                min_value=0, max_value=100, value=30, step=5,
                # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
            )

            selected_val_split = st.slider(
                "Percentage of the Test set to be used as Validation set:",
                min_value=0, max_value=100, value=50, step=5,
                # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
            )

            stratify = st.checkbox("Do Stratified Split?", value=True)

            dataframe_train, dataframe_val, dataframe_test = CommonUtility.split_dataset(
                                                                                            st.session_state["dataframe"],
                                                                                            st.session_state["dataframe_columns"]["anomaly"],
                                                                                            selected_test_split / 100,
                                                                                            selected_val_split / 100,
                                                                                            stratify
            )

            st.session_state["dataframe_train"] = dataframe_train
            st.session_state["dataframe_val"] = dataframe_val
            st.session_state["dataframe_test"] = dataframe_test

            st.markdown("#### Train Dataset")
            st.dataframe(st.session_state["dataframe_train"])
            plot_data_proportions(st.session_state["dataframe_train"])

            num_rows, num_cols = st.session_state["dataframe_train"].shape
            st.write(f"**Number of rows in Train set ::** {num_rows}")
            st.write(f"**Number of columns in Train set ::** {num_cols}")

            st.markdown("#### Validation Dataset")
            st.dataframe(st.session_state["dataframe_val"])
            plot_data_proportions(st.session_state["dataframe_val"])

            num_rows, num_cols = st.session_state["dataframe_val"].shape
            st.write(f"**Number of rows in Validation set ::** {num_rows}")
            st.write(f"**Number of columns in Validation set ::** {num_cols}")

            st.markdown("#### Test Dataset")
            st.dataframe(st.session_state["dataframe_test"])
            plot_data_proportions(st.session_state["dataframe_test"])

            num_rows, num_cols = st.session_state["dataframe_test"].shape
            st.write(f"**Number of rows in Test set ::** {num_rows}")
            st.write(f"**Number of columns in Test set ::** {num_cols}")
            

            st.markdown("---")

    else:
        st.warning("No data uploaded yet! Please go back to the 'Upload Data' page to upload a CSV file.")
        st.markdown("---")
        st.markdown("### How to Use This Page:")
        st.markdown("1. Navigate to the 'Data Uploader' page (usually the first page in the sidebar).")
        st.markdown("2. Upload your CSV file there.")
        st.markdown("3. Once uploaded, return to this 'Exploratory Data Analysis' page to see the analysis.")


if __name__ == "__main__":
    split()