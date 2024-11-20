def similarity_form(selected_imgs):
    """Initializes a form for AI-based image querying with custom input in Streamlit."""
    st.write('Similarity Search')
    with st.form('similarity_form'):
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            st.number_input('limit', min_value=None, max_value=None, value=
                25, label_visibility='collapsed', key='limit')
        with subcol2:
            disabled = not len(selected_imgs)
            st.write('Selected: ', len(selected_imgs))
            st.form_submit_button('Search', disabled=disabled, on_click=
                find_similar_imgs, args=(selected_imgs,))
        if disabled:
            st.error('Select at least one image to search.')
