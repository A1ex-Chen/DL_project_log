def test_name_eq_main(self):
    """Test that the if __name__ == "__main__" block executes without
        error."""
    loader = SourceFileLoader('__main__', os.path.join(os.path.dirname(os.
        path.dirname(__file__)), 'soccertrack', 'logger.py'))
    loader.exec_module(module_from_spec(spec_from_loader(loader.name, loader)))
