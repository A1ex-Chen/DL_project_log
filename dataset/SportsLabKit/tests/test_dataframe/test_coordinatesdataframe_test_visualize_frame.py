def test_visualize_frame(self):
    codf = load_codf(csv_path)
    save_path = outputs_path / 'test_visualize_frame.png'
    if save_path.exists():
        save_path.unlink()
    codf.visualize_frame(0, save_path=save_path)
    assert save_path.exists(), f'File {save_path} does not exist'
