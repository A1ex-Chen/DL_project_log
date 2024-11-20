def test_pipeline_imports(self):
    import diffusers
    import diffusers.pipelines
    all_classes = inspect.getmembers(diffusers, inspect.isclass)
    for cls_name, cls_module in all_classes:
        if hasattr(diffusers.pipelines, cls_name):
            pipeline_folder_module = '.'.join(str(cls_module.__module__).
                split('.')[:3])
            _ = import_module(pipeline_folder_module, str(cls_name))
