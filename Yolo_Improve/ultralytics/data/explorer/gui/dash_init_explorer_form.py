def init_explorer_form():
    """Initializes an Explorer instance and creates embeddings table with progress tracking."""
    datasets = ROOT / 'cfg' / 'datasets'
    ds = [d.name for d in datasets.glob('*.yaml')]
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt',
        'yolov8x.pt', 'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt',
        'yolov8l-seg.pt', 'yolov8x-seg.pt', 'yolov8n-pose.pt',
        'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt',
        'yolov8x-pose.pt']
    with st.form(key='explorer_init_form'):
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox('Select dataset', ds, key='dataset', index=ds.
                index('coco128.yaml'))
        with col2:
            st.selectbox('Select model', models, key='model')
        st.checkbox('Force recreate embeddings', key=
            'force_recreate_embeddings')
        st.form_submit_button('Explore', on_click=_get_explorer)
