# Ultralytics YOLO 🚀, AGPL-3.0 license

import time
from threading import Thread

from ultralytics import Explorer
from ultralytics.utils import ROOT, SETTINGS
from ultralytics.utils.checks import check_requirements

check_requirements(("streamlit>=1.29.0", "streamlit-select>=0.3"))

import streamlit as st
from streamlit_select import image_select














# def persist_reset_form():
#    with st.form("persist_reset"):
#        col1, col2 = st.columns([1, 1])
#        with col1:
#            st.form_submit_button("Reset", on_click=reset)
#
#        with col2:
#            st.form_submit_button("Persist", on_click=update_state, args=("PERSISTING", True))












if __name__ == "__main__":
    layout()