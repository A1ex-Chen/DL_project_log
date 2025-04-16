def test_visualize_frame_with_custom_kwargs(self):
    codf = load_codf(csv_path)
    save_path = (outputs_path /
        'test_visualize_frame_with_custom_save_kwargs.png')
    if save_path.exists():
        save_path.unlink()
    marker_kwargs = {'markerfacecolor': 'green', 'ms': 30}
    saved_kwargs = {'dpi': 300, 'bbox_inches': 'tight'}
    print(codf)
    codf.visualize_frame(0, save_path=save_path, marker_kwargs=
        marker_kwargs, save_kwargs=saved_kwargs)
    assert save_path.exists(), f'File {save_path} does not exist'
