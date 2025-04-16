def query_form():
    """Sets up a form in Streamlit to initialize Explorer with dataset and model selection."""
    with st.form('query_form'):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.text_input('Query',
                "WHERE labels LIKE '%person%' AND labels LIKE '%dog%'",
                label_visibility='collapsed', key='query')
        with col2:
            st.form_submit_button('Query', on_click=run_sql_query)
