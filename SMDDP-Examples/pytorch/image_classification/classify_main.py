def main(args, model_args):
    imgnet_classes = np.array(json.load(open('./LOC_synset_mapping.json', 'r'))
        )
    model = available_models()[args.arch](**model_args.__dict__)
    if args.arch in ['efficientnet-quant-b0', 'efficientnet-quant-b4']:
        check_quant_weight_correctness(model_args.pretrained_from_file, model)
    if not args.cpu:
        model = model.cuda()
    model.eval()
    input = load_jpeg_from_file(args.image, args.image_size, cuda=not args.cpu)
    with torch.no_grad(), autocast(enabled=args.precision == 'AMP'):
        output = torch.nn.functional.softmax(model(input), dim=1)
    output = output.float().cpu().view(-1).numpy()
    top5 = np.argsort(output)[-5:][::-1]
    print(args.image)
    for c, v in zip(imgnet_classes[top5], output[top5]):
        print(f'{c}: {100 * v:.1f}%')
