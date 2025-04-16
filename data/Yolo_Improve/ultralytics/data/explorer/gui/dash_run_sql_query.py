def run_sql_query():
    """Executes an SQL query and returns the results."""
    st.session_state['error'] = None
    query = st.session_state.get('query')
    if query.rstrip().lstrip():
        exp = st.session_state['explorer']
        res = exp.sql_query(query, return_type='arrow')
        st.session_state['imgs'] = res.to_pydict()['im_file']
        st.session_state['res'] = res
