def test_visualize_frames(self):
    codf = load_codf(csv_path)
    save_path = outputs_path / 'test_visualize_frames.mp4'
    if save_path.exists():
        save_path.unlink()
    codf.visualize_frames(save_path=save_path)
    assert save_path.exists(), f'File {save_path} does not exist'
