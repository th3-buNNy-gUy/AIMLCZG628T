import streamlit as st
import pandas as pd
import time
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

from utility.key_generator import KeyGenerator

st.set_page_config(layout="wide")
st.title("CSV Time Series Data Plotter")

st.markdown("Upload a CSV file with a 'datetime' column to visualize its time series data.")

if 'running' not in st.session_state:
    st.session_state.running = False

def stop_stream():
    st.session_state.running = False

def test():
    try:

        if "dataframe" in st.session_state:
            dataframe = st.session_state["dataframe"]

            selected_num_col_scatter_ts = st.selectbox(
                                                        "Select a categorical column for time series:", 
                                                        st.session_state["dataframe_columns"]["numerical"], 
                                                        key=KeyGenerator.generate_key()
            )

            st.subheader("Time Series Data Stream")
            st.session_state.running = st.button("Start Stream")

            # print("A"*1000, st.session_state["dataframe_columns"])

            stop_button = st.button("Stop Stream", on_click=stop_stream)

            if st.session_state.running:
                
                for i in range(len(dataframe)):
                    if not st.session_state.running:
                        break
                    
                    current_df = dataframe.iloc[:i]

                    dataframe_anomaly = current_df.iloc[np.where(current_df[st.session_state["dataframe_columns"]["anomaly"]]!=0)]

                    fig = px.scatter(
                                            dataframe_anomaly, 
                                            x=dataframe_anomaly.index, 
                                            y=selected_num_col_scatter_ts, 
                                            color=st.session_state["dataframe_columns"]["description"],
                                            symbol=st.session_state["dataframe_columns"]["description"],
                    )
                    
                    fig.add_trace(
                                    go.Scatter(
                                                    y=current_df[selected_num_col_scatter_ts],
                                                    x=current_df.index,
                                                    # mode='lines',
                                                    name=selected_num_col_scatter_ts,
                                                    line=dict(color='blue')
                    ))
                    
                    # Wait for 100 milliseconds to simulate a data stream delay
                    # time.sleep(0.1)
                st.plotly_chart(fig, key=991)
                
                if st.session_state.running:
                    st.success("Data stream completed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Entry point for this specific page
if __name__ == "__main__":
    test()
