def utralytics_explorer_docs_callback():
    """Resets the explorer to its initial state by clearing session variables."""
    with st.container(border=True):
        st.image(
            'https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg'
            , width=100)
        st.markdown(
            "<p>This demo is built using Ultralytics Explorer API. Visit <a href='https://docs.ultralytics.com/datasets/explorer/'>API docs</a> to try examples & learn more</p>"
            , unsafe_allow_html=True, help=None)
        st.link_button('Ultrlaytics Explorer API',
            'https://docs.ultralytics.com/datasets/explorer/')
