import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


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

    # Check if a DataFrame is available in session state
    if "dataframe" in st.session_state:
        dataframe = st.session_state["dataframe"]

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                                                    "Describe Your Data", 
                                                    "Dataset", 
                                                    "Univariate Analysis", 
                                                    "Bivariate Analysis", 
                                                    "Multivariate Analysis",
                                                    "Time-series Analysis"
        ])
        
        with tab1:
            dataframe_columns_numerical = dataframe.select_dtypes(include=np.number).columns.tolist()
            with st.container():
                st.subheader("Numerical Columns")
                options = st.multiselect(
                                            "Please select numerical columns",
                                            dataframe_columns_numerical,
                                            default=dataframe_columns_numerical,
                                            key=1
                )
                if len(options) == 0: 
                    st.info("Please select at least one numerical column.")
                else:
                    dataframe_columns_numerical = options
            
            dataframe_columns_anomaly = [col for col in dataframe.select_dtypes(include=np.number).columns.tolist() if not col in dataframe_columns_numerical]
            with st.container():
                st.subheader("Anomaly Column")
                options = st.multiselect(
                                            "Please select the anomaly column",
                                            dataframe_columns_anomaly,
                                            placeholder=None,
                                            max_selections=1,
                                            key=2
                )
                if len(options) == 0: 
                    st.info("Please select the anomaly column.")
                else:
                    dataframe_columns_anomaly = options[0]

            dataframe_columns_dp_description = dataframe.select_dtypes(include="object").columns.tolist()
            with st.container():
                st.subheader("Description Column")
                options = st.multiselect(
                                            "Please select the data point description column",
                                            dataframe_columns_dp_description,
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
            with st.container():
                st.subheader("Overview")
                st.write("Here's a quick look at your data:")
                st.dataframe(dataframe.head())

                st.write(f"**Number of Rows:** {dataframe.shape[0]}")
                st.write(f"**Number of Numerical Columns:** {len(st.session_state['dataframe_columns']['numerical'])}")

            with st.container():
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

            st.markdown("---")


            if st.session_state["dataframe_columns"]["description"] and not isinstance(st.session_state["dataframe_columns"]["description"], list):
                with st.container():
                    st.markdown("#### Data Proportions")
                    
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

                    st.markdown("---")


        with tab3:
            with st.container():
                if st.session_state["dataframe_columns"]["numerical"]:
                    st.markdown("#### Distribution")
                    
                    selected_num_col = st.selectbox("Select a numerical column for histogram:", st.session_state["dataframe_columns"]["numerical"])
                    selected_cat_col = st.selectbox("Select a categorical column for histogram:", [None, st.session_state["dataframe_columns"]["description"]])

                    if selected_num_col:
                        fig_hist = px.histogram(
                                                dataframe, 
                                                x=selected_num_col, 
                                                # nbins=30,
                                                histnorm="percent",
                                                title=f"Distribution of {selected_num_col}",
                                                marginal="violin",
                                                color=selected_cat_col,
                                                height=600,
                                                opacity=0.7,
                                                barmode="overlay",
                                                # barnorm="percent",
                                                log_y=False,
                        ) # Add box plot for distribution
                        st.plotly_chart(fig_hist)


                    st.markdown("#### Central Tendency & Variability")
                    if selected_num_col:
                        data_dict = dict()
                        data_dict["Column"] = list()
                        data_dict["Category"] = list()
                        data_dict["Mean"] = list()
                        data_dict["Median"] = list()
                        data_dict["Minimum Value"] = list()
                        data_dict["Maximum Value"] = list()
                        data_dict["Range"] = list()
                        data_dict["Standard Deviation"] = list()
                        data_dict["Variance"] = list()

                        if selected_cat_col is None:
                            data_dict["Column"].append(selected_num_col)
                            data_dict["Category"].append("All")
                            data_dict["Mean"].append(dataframe[selected_num_col].mean())
                            data_dict["Median"].append(dataframe[selected_num_col].median())
                            data_dict["Minimum Value"].append(dataframe[selected_num_col].min())
                            data_dict["Maximum Value"].append(dataframe[selected_num_col].max())
                            data_dict["Range"].append(dataframe[selected_num_col].max()-dataframe[selected_num_col].min())
                            data_dict["Standard Deviation"].append(dataframe[selected_num_col].std())
                            data_dict["Variance"].append(dataframe[selected_num_col].var())
                        else:
                            tmp = dataframe[selected_num_col]
                            for value in dataframe[selected_cat_col].unique():
                                data_dict["Column"].append(selected_num_col)
                                data_dict["Category"].append(value)
                                data_dict["Mean"].append(tmp.iloc[np.where(dataframe[selected_cat_col]==value)].mean())
                                data_dict["Median"].append(tmp.iloc[np.where(dataframe[selected_cat_col]==value)].median())
                                data_dict["Minimum Value"].append(tmp.iloc[np.where(dataframe[selected_cat_col]==value)].min())
                                data_dict["Maximum Value"].append(tmp.iloc[np.where(dataframe[selected_cat_col]==value)].max())
                                data_dict["Range"].append(tmp.iloc[np.where(dataframe[selected_cat_col]==value)].max()-tmp.iloc[np.where(dataframe[selected_cat_col]==value)].min())
                                data_dict["Standard Deviation"].append(tmp.iloc[np.where(dataframe[selected_cat_col]==value)].std())
                                data_dict["Variance"].append(tmp.iloc[np.where(dataframe[selected_cat_col]==value)].var())

                        data_dict = pd.DataFrame(data_dict, columns=data_dict.keys())
                        data_dict.set_index(["Column", "Category"], inplace=True, drop=True)

                        st.dataframe(data_dict)

                st.markdown("---")


        with tab4:
            with st.container():
                if st.session_state["dataframe_columns"]["numerical"] and len(st.session_state["dataframe_columns"]["numerical"]) > 1:
                    options = st.multiselect(
                                                "Please select numerical columns",
                                                st.session_state["dataframe_columns"]["numerical"],
                                                default=st.session_state["dataframe_columns"]["numerical"],
                                                key=4
                    )


                    selected_cat_col_heatmap = st.selectbox(
                                                                "Select a correlation:", 
                                                                [
                                                                    "Pearson Correlation Coefficient", 
                                                                    "Spearman Rank Correlation",
                                                                    "Kendall Tau Correlation Coefficient",
                                                                ]
                    )    
                    
                    if selected_cat_col_heatmap == "Pearson Correlation Coefficient":
                        corr_matrix = dataframe[options].corr()
                    
                    if selected_cat_col_heatmap == "Spearman Rank Correlation":
                        corr_matrix = dataframe[options].corr(method="spearman")

                    if selected_cat_col_heatmap == "Kendall Tau Correlation Coefficient":
                        corr_matrix = dataframe[options].corr(method="kendall")
                        
                    fig_corr = px.imshow(
                                            corr_matrix,
                                            text_auto=True,
                                            aspect="auto",
                                            color_continuous_scale="Viridis",
                                            # title="Correlation Heatmap of Numerical Columns",
                                            height=1000,
                    )
                    st.plotly_chart(fig_corr)

                st.markdown("---")
                    
            with st.container():
                if st.session_state["dataframe_columns"]["numerical"] and len(st.session_state["dataframe_columns"]["numerical"]) > 1:
                    st.markdown("#### Scatter Plot")
                        
                    dataframe_columns_numerical_scatter = list()
                    options = st.multiselect(
                                                "Please select the data point numerical column",
                                                st.session_state["dataframe_columns"]["numerical"],
                                                default=st.session_state["dataframe_columns"]["numerical"][:2],
                                                max_selections=2,
                                                key=5
                    )
                    if len(options) == 0: 
                        st.info("Please select the description column.")
                    else:
                        dataframe_columns_numerical_scatter = options

                    selected_cat_col_scatter = st.selectbox("Select a categorical column for scatter:", [None, st.session_state["dataframe_columns"]["description"]])
                    
                    fig = None
                    if len(dataframe_columns_numerical_scatter) == 1:
                        fig = px.scatter(
                                            dataframe, 
                                            x=dataframe_columns_numerical_scatter[0], 
                                            y=dataframe_columns_numerical_scatter[0], 
                                            color=selected_cat_col_scatter, 
                                            title="Interactive Scatter Plot",
                                            height=1000,
                                            opacity=0.7,
                        )
                    elif len(dataframe_columns_numerical_scatter) == 2:
                        fig = px.scatter(
                                            dataframe, 
                                            x=dataframe_columns_numerical_scatter[0], 
                                            y=dataframe_columns_numerical_scatter[1], 
                                            color=selected_cat_col_scatter, 
                                            title="Interactive Scatter Plot",
                                            height=1000,
                                            opacity=0.7,
                        )
                    if fig: st.plotly_chart(fig)

                st.markdown("---")


        with tab5:
            with st.container():
                if st.session_state["dataframe_columns"]["numerical"] and len(st.session_state["dataframe_columns"]["numerical"]) > 1:
                    st.markdown("#### Multivariate Scatter Plot")
                        
                    dataframe_columns_numerical_scatter_3d = list()
                    options = st.multiselect(
                                                "Please select the data point numerical column",
                                                st.session_state["dataframe_columns"]["numerical"],
                                                default=st.session_state["dataframe_columns"]["numerical"][:3],
                                                max_selections=3,
                                                key=6
                    )
                    if len(options) == 0: 
                        st.info("Please select the description column.")
                    else:
                        dataframe_columns_numerical_scatter_3d = options

                    selected_cat_col_scatter = st.selectbox(
                                                                "Select a categorical column for scatter:", 
                                                                [None, st.session_state["dataframe_columns"]["description"]], 
                                                                key=7
                    )
                    
                    if len(dataframe_columns_numerical_scatter_3d) == 3:
                        fig = px.scatter_3d(
                                            dataframe, 
                                            x=dataframe_columns_numerical_scatter_3d[0], 
                                            y=dataframe_columns_numerical_scatter_3d[1], 
                                            z=dataframe_columns_numerical_scatter_3d[2], 
                                            color=selected_cat_col_scatter, 
                                            title="Interactive Scatter Plot",
                                            height=1000,
                                            opacity=0.7,
                        )
                        st.plotly_chart(fig)
                    else:
                        st.warning("Need to select 3 numerical columns.")

                st.markdown("---")


        with tab6:
            with st.container():

                selected_num_col_scatter_ts = st.selectbox(
                                                            "Select a categorical column for time series:", 
                                                            st.session_state["dataframe_columns"]["numerical"], 
                                                            key=8
                )

                fig = go.Figure()
                fig.add_trace(
                                go.Scatter(
                                                y=dataframe[selected_num_col_scatter_ts],
                                                x=dataframe.index,
                                                mode='lines',
                                                name=selected_num_col_scatter_ts,
                                                # line=dict(color='blue')
                ))

                selected_cat_col_scatter_ts = st.selectbox(
                                                            "Select a categorical column for scatter:", 
                                                            [None, st.session_state["dataframe_columns"]["description"]], 
                                                            key=9
                )

                if selected_cat_col_scatter_ts:
                    fig.add_trace(
                                    go.Scatter(
                                                    y=dataframe[selected_cat_col_scatter_ts],
                                                    x=dataframe.index,
                                                    mode='markers',
                                                    color=selected_cat_col_scatter_ts, 
                    ))
                st.plotly_chart(fig)

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