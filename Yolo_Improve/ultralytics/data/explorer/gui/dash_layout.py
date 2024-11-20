def layout():
    """Resets explorer session variables and provides documentation with a link to API docs."""
    st.set_page_config(layout='wide', initial_sidebar_state='collapsed')
    st.markdown(
        "<h1 style='text-align: center;'>Ultralytics Explorer Demo</h1>",
        unsafe_allow_html=True)
    if st.session_state.get('explorer') is None:
        init_explorer_form()
        return
    st.button(':arrow_backward: Select Dataset', on_click=reset_explorer)
    exp = st.session_state.get('explorer')
    col1, col2 = st.columns([0.75, 0.25], gap='small')
    imgs = []
    if st.session_state.get('error'):
        st.error(st.session_state['error'])
    elif st.session_state.get('imgs'):
        imgs = st.session_state.get('imgs')
    else:
        imgs = exp.table.to_lance().to_table(columns=['im_file']).to_pydict()[
            'im_file']
        st.session_state['res'] = exp.table.to_arrow()
    total_imgs, selected_imgs = len(imgs), []
    with col1:
        subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
        with subcol1:
            st.write('Max Images Displayed:')
        with subcol2:
            num = st.number_input('Max Images Displayed', min_value=0,
                max_value=total_imgs, value=min(500, total_imgs), key=
                'num_imgs_displayed', label_visibility='collapsed')
        with subcol3:
            st.write('Start Index:')
        with subcol4:
            start_idx = st.number_input('Start Index', min_value=0,
                max_value=total_imgs, value=0, key='start_index',
                label_visibility='collapsed')
        with subcol5:
            reset = st.button('Reset', use_container_width=False, key='reset')
            if reset:
                st.session_state['imgs'] = None
                st.experimental_rerun()
        query_form()
        ai_query_form()
        if total_imgs:
            labels, boxes, masks, kpts, classes = None, None, None, None, None
            task = exp.model.task
            if st.session_state.get('display_labels'):
                labels = st.session_state.get('res').to_pydict()['labels'][
                    start_idx:start_idx + num]
                boxes = st.session_state.get('res').to_pydict()['bboxes'][
                    start_idx:start_idx + num]
                masks = st.session_state.get('res').to_pydict()['masks'][
                    start_idx:start_idx + num]
                kpts = st.session_state.get('res').to_pydict()['keypoints'][
                    start_idx:start_idx + num]
                classes = st.session_state.get('res').to_pydict()['cls'][
                    start_idx:start_idx + num]
            imgs_displayed = imgs[start_idx:start_idx + num]
            selected_imgs = image_select(f'Total samples: {total_imgs}',
                images=imgs_displayed, use_container_width=False, labels=
                labels, classes=classes, bboxes=boxes, masks=masks if task ==
                'segment' else None, kpts=kpts if task == 'pose' else None)
    with col2:
        similarity_form(selected_imgs)
        st.checkbox('Labels', value=False, key='display_labels')
        utralytics_explorer_docs_callback()
