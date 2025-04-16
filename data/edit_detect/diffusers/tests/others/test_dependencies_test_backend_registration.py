def test_backend_registration(self):
    import diffusers
    from diffusers.dependency_versions_table import deps
    all_classes = inspect.getmembers(diffusers, inspect.isclass)
    for cls_name, cls_module in all_classes:
        if 'dummy_' in cls_module.__module__:
            for backend in cls_module._backends:
                if backend == 'k_diffusion':
                    backend = 'k-diffusion'
                elif backend == 'invisible_watermark':
                    backend = 'invisible-watermark'
                assert backend in deps, f'{backend} is not in the deps table!'
