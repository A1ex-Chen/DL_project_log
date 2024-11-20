def run_ai_query():
    """Execute SQL query and update session state with query results."""
    if not SETTINGS['openai_api_key']:
        st.session_state['error'] = (
            'OpenAI API key not found in settings. Please run yolo settings openai_api_key="..."'
            )
        return
    import pandas
    st.session_state['error'] = None
    query = st.session_state.get('ai_query')
    if query.rstrip().lstrip():
        exp = st.session_state['explorer']
        res = exp.ask_ai(query)
        if not isinstance(res, pandas.DataFrame) or res.empty:
            st.session_state['error'] = (
                'No results found using AI generated query. Try another query or rerun it.'
                )
            return
        st.session_state['imgs'] = res['im_file'].to_list()
        st.session_state['res'] = res
