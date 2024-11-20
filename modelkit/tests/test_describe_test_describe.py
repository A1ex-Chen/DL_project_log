def test_describe(monkeypatch):
    monkeypatch.setenv('MODELKIT_ASSETS_DIR', os.path.join(TEST_DIR,
        'testdata', 'test-bucket'))


    class SomeSimpleValidatedModelWithAsset(Model[str, str]):
        """
        This is a summary

        that also has plenty more text
        """
        CONFIGURATIONS = {'some_model_a': {'asset': 'assets-prefix'}}

        def _predict(self, item):
            return item


    class SomeSimpleValidatedModelA(Model[str, str]):
        """
        This is a summary

        that also has plenty more text
        """
        CONFIGURATIONS = {'some_model_a': {}}

        def _predict(self, item):
            return item


    class ItemModel(pydantic.BaseModel):
        string: str


    class ResultModel(pydantic.BaseModel):
        sorted: str


    class A:

        def __init__(self):
            self.x = 1
            self.y = 2


    class SomeComplexValidatedModelA(Model[ItemModel, ResultModel]):
        """
        More complex

        With **a lot** of documentation
        """
        CONFIGURATIONS = {'some_complex_model_a': {'model_dependencies': [
            'some_model_a'], 'asset': os.path.join(TEST_DIR, 'testdata',
            'test-bucket', 'assets-prefix', 'category', 'asset', '0.0'),
            'model_settings': {'batch_size': 128}}}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.some_object = A()

        def _predict(self, item):
            return item
    library = ModelLibrary()
    library.describe()
    library = ModelLibrary(models=[SomeSimpleValidatedModelA,
        SomeSimpleValidatedModelWithAsset, SomeComplexValidatedModelA])
    library.describe()
    library = ModelLibrary(models=[SomeSimpleValidatedModelA,
        SomeComplexValidatedModelA])
    console = Console(no_color=True, force_terminal=False, width=130)
    with console.capture() as capture:
        library.describe(console=console)
    if platform.system() == 'Windows' or sys.version_info[:2] < (3, 11):
        return
    r = ReferenceText(os.path.join(TEST_DIR, 'testdata'))
    captured = capture.get()
    EXCLUDED = ['load time', 'load memory', 'asset', 'category/asset', os.
        path.sep]
    captured = '\n'.join(line for line in captured.split('\n') if not any(x in
        line for x in EXCLUDED))
    r.assert_equal('library_describe.txt', captured)
