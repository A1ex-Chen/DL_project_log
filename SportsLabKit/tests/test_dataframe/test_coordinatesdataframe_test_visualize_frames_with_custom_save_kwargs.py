def test_visualize_frames_with_custom_save_kwargs(self):
    codf = load_codf(csv_path)
    save_path = (outputs_path /
        'test_visualize_frames_with_custom_save_kwargs.mp4')
    if save_path.exists():
        save_path.unlink()
    saved_kwargs = {'dpi': 300, 'fps': 50}
    codf.visualize_frames(save_path=save_path, save_kwargs=saved_kwargs)
    assert save_path.exists(), f'File {save_path} does not exist'
