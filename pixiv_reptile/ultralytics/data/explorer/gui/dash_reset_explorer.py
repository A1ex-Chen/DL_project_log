def reset_explorer():
    """Resets the explorer to its initial state by clearing session variables."""
    st.session_state['explorer'] = None
    st.session_state['imgs'] = None
    st.session_state['error'] = None
