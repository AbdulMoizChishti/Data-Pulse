from operator import index
import streamlit as st
import plotly.express as px
import pycaret
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling
from pydantic import BaseModel
from pydantic_settings import *
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Data-Pulse")
    choice = st.radio("Navigation", ["Upload","Profiling"])
    st.info("This project application helps you build and explore your data.")


