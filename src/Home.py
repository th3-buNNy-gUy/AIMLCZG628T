import streamlit as st

st.set_page_config(
    page_title="Anomaly Detection Application",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)


with st.container():
    cols = st.columns(5)

    if cols[-1].button("Upload Data page >>", use_container_width=True, type="tertiary"):
        st.switch_page("pages/1_Upload_Data.py")


def main():
    """
    Main function for the Home page, providing an overview of the application's capabilities.
    """

    st.title("Welcome to Your Interactive Data Science Application!")
    st.write("This application is designed to help you quickly upload, explore, analyze, and model your datasets with ease. Leveraging the power of Python and Streamlit, it provides an intuitive interface for various data science tasks.")
    st.markdown("---")

    st.header("Application Capabilities:")

    st.markdown("""
    ### 1. Data Upload (CSV Format) ⬆️
    Easily upload your datasets in CSV format. The application will display a preview of your data, allowing you to quickly verify its structure and content.
    * **Go to:** `Upload Data` in the sidebar.
    """)
    st.page_link("pages/1_Upload_Data.py", label="Page 1", icon="1️⃣")

    st.markdown("""
    ### 2. Exploratory Data Analysis (EDA) 📊
    Dive deep into your data with comprehensive exploratory data analysis tools. Visualize distributions, identify missing values, understand correlations, and gain insights into your dataset's characteristics.
    * **Go to:** `Exploratory Data Analysis` in the sidebar.
    """)
    if st.button("Page 2"):
        st.switch_page("pages/2_Exploratory_Data_Analysis.py")

    st.markdown("""
    ### 3. Statistical Tests 🧪
    (Coming Soon!) Perform various statistical tests to validate hypotheses, compare groups, and understand relationships within your data. This section will help you draw robust conclusions from your observations.
    * **Go to:** `Statistical Tests` in the sidebar.
    """)

    st.markdown("""
    ### 4. Statistical Modeling 🧠
    (Coming Soon!) Build and evaluate statistical models on your prepared data. This module will support different modeling techniques to predict outcomes, classify data, or uncover hidden patterns.
    * **Go to:** `Statistical Modeling` in the sidebar.
    """)

    st.markdown("---")
    st.info("To get started, please navigate to the 'Upload Data' page using the sidebar on the left.")

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()