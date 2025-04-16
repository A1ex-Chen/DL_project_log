def main(args, model_args, model_arch):
    quant_arch = args.arch in ['efficientnet-quant-b0', 'efficientnet-quant-b4'
        ]
    if quant_arch:
        (pytorch_quantization.nn.modules.tensor_quantizer.TensorQuantizer.
            use_fb_fake_quant) = True
    model = model_arch(**model_args.__dict__)
    if quant_arch and model_args.pretrained_from_file is not None:
        check_quant_weight_correctness(model_args.pretrained_from_file, model)
    image_size = (args.image_size if args.image_size is not None else model
        .arch.default_image_size)
    train_loader = get_dataloader(image_size, args.batch_size, model_args.
        num_classes)
    inputs = prepare_inputs(train_loader, args.device)
    final_model_path = args.output if args.output is not None else final_name(
        model_args.pretrained_from_file)
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, inputs[0], final_model_path, verbose=True,
            opset_version=13, enable_onnx_checker=True, do_constant_folding
            =True)
