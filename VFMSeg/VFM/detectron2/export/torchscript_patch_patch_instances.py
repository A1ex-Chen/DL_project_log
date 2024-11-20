@contextmanager
def patch_instances(fields):
    """
    A contextmanager, under which the Instances class in detectron2 is replaced
    by a statically-typed scriptable class, defined by `fields`.
    See more in `scripting_with_instances`.
    """
    with tempfile.TemporaryDirectory(prefix='detectron2'
        ) as dir, tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
        suffix='.py', dir=dir, delete=False) as f:
        try:
            _clear_jit_cache()
            cls_name, s = _gen_instance_module(fields)
            f.write(s)
            f.flush()
            f.close()
            module = _import(f.name)
            new_instances = getattr(module, cls_name)
            _ = torch.jit.script(new_instances)
            Instances.__torch_script_class__ = True
            Instances._jit_override_qualname = (torch._jit_internal.
                _qualified_name(new_instances))
            _add_instances_conversion_methods(new_instances)
            yield new_instances
        finally:
            try:
                del Instances.__torch_script_class__
                del Instances._jit_override_qualname
            except AttributeError:
                pass
            sys.modules.pop(module.__name__)
