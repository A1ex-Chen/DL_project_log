def _test_from_save_pretrained_dynamo(in_queue, out_queue, timeout):
    error = None
    try:
        init_dict, model_class = in_queue.get(timeout=timeout)
        model = model_class(**init_dict)
        model.to(torch_device)
        model = torch.compile(model)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)
        assert new_model.__class__ == model_class
    except Exception:
        error = f'{traceback.format_exc()}'
    results = {'error': error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()
