import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import uuid

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)


with st.container():
    cols = st.columns(5)
    
    if cols[0].button("<< Upload Data page", use_container_width=True, type="tertiary"):
        st.switch_page("pages/1_Upload_Data.py")
        
    # if cols[-1].button("Exploratory Data Analysis >>", use_container_width=True, type="tertiary"):
    #     st.switch_page("pages/2_Exploratory_Data_Analysis.py")

def eda_page():
    """
    Function to display the Exploratory Data Analysis page.
    It retrieves the DataFrame from session state and performs basic EDA.
    """

    st.title("Exploratory Data Analysis")
    st.write("Perform basic exploratory data analysis on your uploaded dataset.")
    st.markdown("---")

    # Check if a DataFrame is available in session state
    if "dataframe" in st.session_state:
        dataframe = st.session_state["dataframe"]

        tab1, tab2, tab3 = st.tabs(["Describe Your Data", "Dataset", "Owl"])
        
        with tab1:
            with st.container():
                st.subheader("Numerical Columns")
                options = st.multiselect(
                                            "Please select numerical columns",
                                            dataframe.columns,
                                            default=dataframe.columns,
                                            key=1
                )
                if len(options) == 0: 
                    st.info("Please select at least one numerical column.")
                else:
                    dataframe_columns_numerical = options

            with st.container():
                st.subheader("Anomaly Column")
                options = st.multiselect(
                                            "Please select the anomaly column",
                                            dataframe.columns,
                                            placeholder=None,
                                            max_selections=1,
                                            key=2
                )
                if len(options) == 0: 
                    st.info("Please select the anomaly column.")
                else:
                    dataframe_columns_anomaly = options[0]

            with st.container():
                st.subheader("Description Column")
                options = st.multiselect(
                                            "Please select the data point description column",
                                            dataframe.columns,
                                            placeholder=None,
                                            max_selections=1,
                                            key=3
                )
                if len(options) == 0: 
                    st.info("Please select the description column.")
                else:
                    dataframe_columns_dp_description = options[0]

            st.session_state["dataframe_columns"] = {
                                                        "numerical" : dataframe_columns_numerical,
                                                        "anomaly" : dataframe_columns_anomaly,
                                                        "description" : dataframe_columns_dp_description
            }

        with tab2:
            st.subheader("Overview")
            st.write("Here's a quick look at your data:")
            st.dataframe(dataframe.head())

            st.write(f"**Number of Rows:** {dataframe.shape[0]}")
            st.write(f"**Number of Columns:** {dataframe.shape[1]}")

            st.subheader("Column Information")
            st.write("Data types and non-null counts for each column:")
            # Create a DataFrame for info display
            info_df = pd.DataFrame({
                "Column": dataframe.columns,
                "Data Type": dataframe.dtypes,
                "Non-Null Count": dataframe.count(),
                "Missing Values": dataframe.isnull().sum(),
                "Missing %": (dataframe.isnull().sum() / len(dataframe) * 100).round(2)
            })
            st.dataframe(info_df)

            st.subheader("Descriptive Statistics")
            st.write("Summary statistics for numerical columns:")
            st.dataframe(dataframe.describe())

            st.subheader("Missing Values Visualization")
            # Create a simple bar chart for missing values
            missing_data = dataframe.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            if not missing_data.empty:
                fig_missing = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    labels={"x": "Column", "y": "Number of Missing Values"},
                    title="Missing Values per Column"
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.info("No missing values found in the dataset!")

            st.markdown("---")
            st.subheader("Univariate Analysis")

            # Separate numerical and categorical columns
            numerical_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = dataframe.select_dtypes(include="object").columns.tolist()

            if numerical_cols:
                st.markdown("#### Numerical Columns")
                selected_num_col = st.selectbox("Select a numerical column for histogram:", numerical_cols)
                if selected_num_col:
                    fig_hist = px.histogram(dataframe, x=selected_num_col, nbins=30,
                                            title=f"Distribution of {selected_num_col}",
                                            marginal="box") # Add box plot for distribution
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.write(f"**Unique values in {selected_num_col}:** {dataframe[selected_num_col].nunique()}")
                    st.write(f"**Mean:** {dataframe[selected_num_col].mean():.2f}")
                    st.write(f"**Median:** {dataframe[selected_num_col].median():.2f}")
                    st.write(f"**Standard Deviation:** {dataframe[selected_num_col].std():.2f}")

            if categorical_cols:
                st.markdown("#### Categorical Columns")
                selected_cat_col = st.selectbox("Select a categorical column for value counts and bar chart:", categorical_cols)
                if selected_cat_col:
                    st.write(f"**Value Counts for {selected_cat_col}:**")
                    value_counts = dataframe[selected_cat_col].value_counts().reset_index()
                    value_counts.columns = [selected_cat_col, "Count"]
                    st.dataframe(value_counts)

                    fig_bar = px.bar(value_counts, x=selected_cat_col, y="Count",
                                    title=f"Value Counts of {selected_cat_col}")
                    st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            st.subheader("Bivariate Analysis (Correlation Heatmap)")
            if numerical_cols and len(numerical_cols) > 1:
                corr_matrix = dataframe[numerical_cols].corr()
                fig_corr = px.imshow(corr_matrix,
                                    text_auto=True,
                                    aspect="auto",
                                    color_continuous_scale="Viridis",
                                    title="Correlation Heatmap of Numerical Columns")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least two numerical columns to display a correlation heatmap.")

        with tab3:
            st.header("An owl")
            st.image("https://static.streamlit.io/examples/owl.jpg", width=200)



    else:
        st.warning("No data uploaded yet! Please go back to the 'Upload Data' page to upload a CSV file.")
        st.markdown("---")
        st.markdown("### How to Use This Page:")
        st.markdown("1. Navigate to the 'Data Uploader' page (usually the first page in the sidebar).")
        st.markdown("2. Upload your CSV file there.")
        st.markdown("3. Once uploaded, return to this 'Exploratory Data Analysis' page to see the analysis.")

# Entry point for this specific page
if __name__ == "__main__":
    eda_page()