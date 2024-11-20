def test_generate_model_card_with_library_name(self):
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'README.md'
        file_path.write_text('---\nlibrary_name: foo\n---\nContent\n')
        model_card = load_or_create_model_card(file_path)
        populate_model_card(model_card)
        assert model_card.data.library_name == 'foo'
