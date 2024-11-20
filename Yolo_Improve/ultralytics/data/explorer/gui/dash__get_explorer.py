def _get_explorer():
    """Initializes and returns an instance of the Explorer class."""
    exp = Explorer(data=st.session_state.get('dataset'), model=st.
        session_state.get('model'))
    thread = Thread(target=exp.create_embeddings_table, kwargs={'force': st
        .session_state.get('force_recreate_embeddings')})
    thread.start()
    progress_bar = st.progress(0, text='Creating embeddings table...')
    while exp.progress < 1:
        time.sleep(0.1)
        progress_bar.progress(exp.progress, text=
            f'Progress: {exp.progress * 100}%')
    thread.join()
    st.session_state['explorer'] = exp
    progress_bar.empty()
