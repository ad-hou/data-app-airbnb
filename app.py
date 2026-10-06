"""Point d'entree : configuration, style et navigation."""
import streamlit as st

from src.style import apply_style

st.set_page_config(page_title="Airbnb Paris", page_icon=":material/home_work:", layout="wide")
apply_style()

pages = [
    st.Page("views/exploration.py", title="Exploration", icon=":material/insights:", default=True),
    st.Page("views/modele.py", title="Modele", icon=":material/model_training:"),
    st.Page("views/prediction.py", title="Prediction", icon=":material/payments:"),
    st.Page("views/a_propos.py", title="A propos", icon=":material/info:"),
]
st.navigation(pages).run()
