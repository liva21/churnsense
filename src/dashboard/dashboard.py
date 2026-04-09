import streamlit as st
import requests

st.set_page_config(page_title="ChurnSense Dashboard", layout="wide")

st.title("ChurnSense Dashboard")
st.write("Welcome to the ChurnSense customer churn prediction dashboard.")

# Example API call
# response = requests.get("http://localhost:8000/")
# if response.status_code == 200:
#     st.success(f"API is running: {response.json()}")
