import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from models.dimensionality_reduction import principle_component_analysis, linear_discriminant_analysis, non_negative_matrix_factorization
from models.dimensionality_reduction import t_distributed_stochastic_neighbor_embedding, uniform_manifold_approximation_and_projection

st.set_page_config(
    page_title="Dimensionality Reduction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     "Get Help": "https://www.extremelycoolapp.com/help",
    #     "Report a bug": "https://www.extremelycoolapp.com/bug",
    #     "About": "# This is a header. This is an *extremely* cool app!"
    # }
)


def dr_page():

    st.title("Dimensionality Reduction")
    st.write("Perform dimensionality reduction on your uploaded dataset.")

    if "dataframe" in st.session_state:

        with st.container():
            dr_techniques = {
                                "Principle Component Analysis":principle_component_analysis,
                                # "Linear Discriminant Analysis":linear_discriminant_analysis,
                                "Non-Negative Matrix Factorization":non_negative_matrix_factorization,
                                "t-Distributed Stochastic Neighbor Embedding":t_distributed_stochastic_neighbor_embedding,
                                "Uniform Manifold Approximation and Projection":uniform_manifold_approximation_and_projection,
            }

            selected_dr_method = st.selectbox(
                                                "Select a dimensionality reduction technique:", 
                                                dr_techniques.keys(), 
                                                key=11
            )
            selected_n_components = None
            if selected_dr_method != "Linear Discriminant Analysis":
                selected_n_components = st.slider(
                    "Number of Components for the Dimensionality Reduction technique:",
                    min_value=2, max_value=len(st.session_state["dataframe_columns"]["numerical"]), step=1,
                    # help="Points with a Z-score (standard deviations from mean) above this threshold will be marked as anomalies."
                )

            dr_dataframe = dr_techniques[selected_dr_method](
                                                                st.session_state["dataframe"], 
                                                                st.session_state["dataframe_columns"]["anomaly"], 
                                                                st.session_state["dataframe_columns"]["description"],
                                                                selected_n_components
            )

            st.dataframe(dr_dataframe)

            st.markdown("---")

        with st.container():
            st.markdown("#### Scatter Plot")

            component_columns = dr_dataframe.drop([st.session_state["dataframe_columns"]["anomaly"], st.session_state["dataframe_columns"]["description"]], axis=1).columns
                
            dataframe_columns_numerical_scatter = list()
            options = st.multiselect(
                                        "Please select the data point numerical column",
                                        component_columns,
                                        default=component_columns[:2],
                                        max_selections=2,
                                        key=5
            )
            if len(options) == 0: 
                st.info("Please select the description column.")
            else:
                dataframe_columns_numerical_scatter = options

            selected_cat_col_scatter = st.selectbox("Select a categorical column for scatter:", [None, st.session_state["dataframe_columns"]["description"]])
            
            fig = None
            fig = px.scatter(
                                dr_dataframe, 
                                x=dataframe_columns_numerical_scatter[0], 
                                y=dataframe_columns_numerical_scatter[1 if len(dataframe_columns_numerical_scatter) == 2 else 0], 
                                color=selected_cat_col_scatter, 
                                symbol=selected_cat_col_scatter,
                                title="Interactive Scatter Plot",
                                height=1000,
                                opacity=0.7,
            )
            st.plotly_chart(fig)

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
    dr_page()