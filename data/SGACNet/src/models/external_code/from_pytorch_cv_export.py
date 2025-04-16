def export(model, rgb, onnx_name, opset=10):
    onnx_file_path = os.path.join(out_dir, onnx_name)
    torch.onnx.export(model, rgb, onnx_file_path, export_params=True,
        input_names=['rgb'], output_names=['output'], do_constant_folding=
        True, verbose=False, opset_version=opset)
    print(f'exported {onnx_name}')
