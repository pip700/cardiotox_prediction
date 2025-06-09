# project/Home.py

import streamlit as st
from pages.utils import set_global_configs # Import the utility to set configs once

# Set global configurations (seeds, deterministic algorithms, matplotlib style)
set_global_configs()

st.set_page_config(
    page_title="Multi-Ion Channel Blocker Predictor",
    page_icon="💊",
    layout="wide"
)

st.title("Welcome to the Cardiotoxicity Prediction App!")

st.markdown("""
    This application allows you to predict whether a given molecule is a blocker for
    different types of ion channels: **hERG (Kav), Cav, and Nav**.

    Use the sidebar to navigate to the specific predictor you wish to use.

    ### How to Use:
    1.  **Select a Predictor:** Choose 'KAV Predictor', 'CAV Predictor', or 'NAV Predictor' from the sidebar on the left.
    2.  **Input Molecules:** On the selected predictor page, you can either:
        *   Enter a **single SMILES string** for an immediate prediction.
        *   **Upload a CSV file** containing a list of SMILES strings for batch prediction.
    3.  **Get Predictions:** Click the "Predict" button to see the results, including prediction score, class, and molecular importance visualizations.
    4.  **Download Results:** Download the predictions and importance data as CSV files.

    ---
    **About the Models:**
    *   **KAV (hERG)**: Predicts hERG potassium channel blockers.
    *   **CAV (Calcium Channels)**: Predicts Cav channel blockers.
    *   **NAV (Sodium Channels)**: Predicts Nav channel blockers.
""")

st.info("Please select a predictor from the sidebar to begin!")
