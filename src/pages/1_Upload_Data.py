import streamlit as st
import pandas as pd
import io
import numpy as np

st.set_page_config(
    page_title="CSV File Uploader & Viewer",
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

    if cols[0].button("<< Home page", use_container_width=True, type="tertiary"):
        st.switch_page("Home.py")
        
    if cols[-1].button("Exploratory Data Analysis page >>", use_container_width=True, type="tertiary"):
        st.switch_page("pages/2_Exploratory_Data_Analysis.py")


def upload_page():
    """
    Main function to run the Streamlit application.
    It sets up the page configuration, handles file uploads,
    and displays the CSV content as a DataFrame.
    """

    # Display the main title and a brief description for the homepage
    st.title("CSV File Uploader & Viewer")
    st.write("Welcome! This application allows you to upload a CSV file and view its contents directly in your browser as a Pandas DataFrame.")
    st.markdown("---") # Add a separator for better visual organization

    # Create a file uploader widget
    # 'type="csv"' restricts the upload to CSV files only
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Select a CSV file from your local machine.")

    # Check if a file has been uploaded by the user
    if uploaded_file is not None:
        # st.success("CSV file uploaded successfully!")
        st.toast("CSV file uploaded successfully!")
        st.markdown("---") # Add a separator for better visual organization
        
        try:
            with st.container():
                st.subheader("Data Preview ::")

                # Read the CSV file into a pandas DataFrame
                # io.StringIO is used to treat the uploaded file (bytes) as a string for pandas
                dataframe = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
                dataframe.set_index("datetime", inplace=True, drop=True)

                # Display the DataFrame using st.dataframe, which provides interactive features
                st.dataframe(dataframe)
                st.session_state["dataframe"] = dataframe

                st.markdown("---")


            with st.container():
                st.subheader("Basic Statistics ::")

                # Display descriptive statistics of the numerical columns
                st.write(dataframe.describe())

                st.markdown("---")


            with st.container():
                st.subheader("Data Information ::")

                # Display information about columns, including data types and non-null values
                # Using st.write(dataframe.info()) directly doesn't work well for display;
                # A better way is to iterate or convert info to a string or table.
                # For simplicity, let's show dtypes and shape.
                st.write(f"**Number of rows:** {dataframe.shape[0]}")
                st.write(f"**Number of columns:** {dataframe.shape[1]}")

                st.write("**Column Data Types:**")
                st.write(dataframe.dtypes)

                st.markdown("---")

            # st.success("CSV file successfully loaded and displayed!")
            st.toast("CSV file successfully loaded and displayed!")
            st.info("Now, head over to the 'Exploratory Data Analysis' page in the sidebar to start analyzing your data!")

        except Exception as e:
            # Catch any errors during file processing (e.g., malformed CSV)
            st.error(f"An error occurred while reading the CSV file: {e}")
            st.warning("Please ensure the uploaded file is a valid CSV format and not corrupted.")
    else:
        # Message displayed when no file is uploaded yet
        st.info("Please upload a CSV file using the button above to get started.")
        st.markdown("---")
        st.markdown("### How to Use:")
        st.markdown("1. Click on the 'Choose a CSV file' button.")
        st.markdown("2. Select a `.csv` file from your computer.")
        st.markdown("3. The application will automatically display a preview of your data.")
        st.markdown("4. Once uploaded, use the sidebar to navigate to 'Exploratory Data Analysis'.")

# Entry point for the Streamlit application
if __name__ == "__main__":
    upload_page()