def ai_query_form():
    """Sets up a Streamlit form for user input to initialize Explorer with dataset and model selection."""
    with st.form('ai_query_form'):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.text_input('Query', 'Show images with 1 person and 1 dog',
                label_visibility='collapsed', key='ai_query')
        with col2:
            st.form_submit_button('Ask AI', on_click=run_ai_query)
