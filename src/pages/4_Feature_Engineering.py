import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from page_data.feature_engineering import InteractionFeatures, DifferenceFeatures, EfficiencyPerformanceMetrics, HigherOrderInteractiveFeatures, apply_feature_engineering

from utility.key_generator import KeyGenerator

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)

def fe_page():
        
    st.title("Feature Engineering")
    st.write("Perform feature engineering on your dataset.")

    if "dataframe" in st.session_state:
        dataframe = st.session_state["dataframe"]
        
        st.subheader("Feature Engineering Options")
        st.write("Select the feature engineering techniques you want to apply to your dataset.")
        st.markdown("---") # Add a separator for better visual organization

        with st.container():
            st.subheader(InteractionFeatures.name)

            features = InteractionFeatures.feature_list
            data = {
                'Checkbox': [True] * len(features),  # Initializes all checkboxes to False
                'Name': [item.name for item in features],
                'Description': [item.description for item in features],
            }

            df = pd.DataFrame(data)
            row_height = 100
            header_offset = 38
            calculated_height = (row_height * len(df)) + header_offset
            
            edited_df1 = st.data_editor(
                                            df,
                                            hide_index=True,
                                            key=401,
                                            use_container_width=True,
                                            row_height=row_height,
                                            column_config={
                                                "Checkbox": st.column_config.CheckboxColumn(
                                                                                                "Select",
                                                                                                help="Select the line item",
                                                                                                default=True,
                                                                                                width="small",
                                                )
                                            },
                                            height=calculated_height,
                                            disabled=("Name", "Description") # Disable editing for these columns
            )

        with st.container():
            st.subheader(DifferenceFeatures.name)

            features = DifferenceFeatures.feature_list
            data = {
                'Checkbox': [True] * len(features),  # Initializes all checkboxes to False
                'Name': [item.name for item in features],
                'Description': [item.description for item in features],
            }

            df = pd.DataFrame(data)
            row_height = 100
            header_offset = 38
            calculated_height = (row_height * len(df)) + header_offset
            
            edited_df2 = st.data_editor(
                                            df,
                                            key=402,
                                            hide_index=True,
                                            use_container_width=True,
                                            row_height=row_height,
                                            column_config={
                                                "Checkbox": st.column_config.CheckboxColumn(
                                                                                                "Select",
                                                                                                help="Select the line item",
                                                                                                default=True,
                                                                                                width="small",
                                                )
                                            },
                                            height=calculated_height,
                                            disabled=("Name", "Description") # Disable editing for these columns
            )

        with st.container():
            st.subheader(EfficiencyPerformanceMetrics.name)

            features = EfficiencyPerformanceMetrics.feature_list
            data = {
                'Checkbox': [True] * len(features),  # Initializes all checkboxes to False
                'Name': [item.name for item in features],
                'Description': [item.description for item in features],
            }

            df = pd.DataFrame(data)
            row_height = 100
            header_offset = 38
            calculated_height = (row_height * len(df)) + header_offset
            
            edited_df3 = st.data_editor(
                                            df,
                                            key=403,
                                            hide_index=True,
                                            use_container_width=True,
                                            row_height=row_height,
                                            column_config={
                                                "Checkbox": st.column_config.CheckboxColumn(
                                                                                                "Select",
                                                                                                help="Select the line item",
                                                                                                default=True,
                                                                                                width="small",
                                                )
                                            },
                                            height=calculated_height,
                                            disabled=("Name", "Description") # Disable editing for these columns
            )

        with st.container():
            st.subheader(HigherOrderInteractiveFeatures.name)

            features = HigherOrderInteractiveFeatures.feature_list
            data = {
                'Checkbox': [True] * len(features),  # Initializes all checkboxes to False
                'Name': [item.name for item in features],
                'Description': [item.description for item in features],
            }

            df = pd.DataFrame(data)
            row_height = 100
            header_offset = 38
            calculated_height = (row_height * len(df)) + header_offset
            
            edited_df4 = st.data_editor(
                                            df,
                                            key=404,
                                            hide_index=True,
                                            use_container_width=True,
                                            row_height=row_height,
                                            column_config={
                                                "Checkbox": st.column_config.CheckboxColumn(
                                                                                                "Select",
                                                                                                help="Select the line item",
                                                                                                default=True,
                                                                                                width="small",
                                                )
                                            },
                                            height=calculated_height,
                                            disabled=("Name", "Description") # Disable editing for these columns
            )

        st.markdown("---") # Add a separator for better visual organization
        with st.container():

            selected_features = list()

            for index, row in edited_df1.iterrows():
                if row['Checkbox']:
                    selected_features.append(InteractionFeatures.feature_list[index])

            for index, row in edited_df2.iterrows():
                if row['Checkbox']:
                    selected_features.append(DifferenceFeatures.feature_list[index])

            for index, row in edited_df3.iterrows():
                if row['Checkbox']:
                    selected_features.append(EfficiencyPerformanceMetrics.feature_list[index])

            for index, row in edited_df4.iterrows():
                if row['Checkbox']:
                    selected_features.append(HigherOrderInteractiveFeatures.feature_list[index])

            dataset_name = st.text_input("Enter the name of the dataset", placeholder="e.g. Classification Dataset").strip()
            is_applied = st.button(
                                    "Apply", 
                                    use_container_width=False, 
                                    type="secondary", 
                                    on_click=apply_feature_engineering, args=(selected_features, dataframe),
            )

        if is_applied: 
            with st.container():
                st.subheader("Data Preview ::")
                st.dataframe(dataframe)

                st.subheader("Basic Statistics ::")
                st.dataframe(dataframe.describe())

                st.subheader("Data Information ::")
                st.write(f"**Number of rows:** {dataframe.shape[0]}")
                st.write(f"**Number of columns:** {dataframe.shape[1]}")
                st.write("**Column Data Types:**")
                st.write(dataframe.dtypes)
                ## Convert the DataFrame to a downloadable format (e.g., CSV)
                ## It's recommended to use caching for efficiency, especially with large DataFrames
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=True).encode('utf-8')

                # Create the download button with a specified file_name
                st.download_button(
                                        label="Download",
                                        data=convert_df_to_csv(dataframe),
                                        file_name=f"{dataset_name}.csv",  # This sets the downloaded file's name
                                        mime="text/csv",
                )

    else:
        st.success("Please upload a dataset in the 'Upload Data' page.")
        st.info("No dataset found in session state.")

# Entry point for this specific page
if __name__ == "__main__":
    fe_page()