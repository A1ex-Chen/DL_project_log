def import_flatten_impl():
    global flatten_impl, unflatten_impl, imported_flatten_impl
    try:
        import apex_C
        flatten_impl = apex_C.flatten
        unflatten_impl = apex_C.unflatten
    except ImportError:
        print(
            'Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten.'
            )
        flatten_impl = torch._utils._flatten_dense_tensors
        unflatten_impl = torch._utils._unflatten_dense_tensors
    imported_flatten_impl = True
