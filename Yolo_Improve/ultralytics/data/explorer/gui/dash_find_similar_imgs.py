def find_similar_imgs(imgs):
    """Initializes a Streamlit form for AI-based image querying with custom input."""
    exp = st.session_state['explorer']
    similar = exp.get_similar(img=imgs, limit=st.session_state.get('limit'),
        return_type='arrow')
    paths = similar.to_pydict()['im_file']
    st.session_state['imgs'] = paths
    st.session_state['res'] = similar
