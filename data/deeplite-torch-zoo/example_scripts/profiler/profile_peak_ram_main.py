def main(opt):
    model = get_model(model_name=opt.model, dataset_name=opt.dataset,
        num_classes=opt.num_classes, pretrained=opt.pretrained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    img_size = 3, opt.img_size, opt.img_size
    input_tensor = torch.randn(1, *img_size).to(device)
    max_ram_usage = profile_ram(model, input_tensor, num_bytes=opt.num_bytes)
    print(f'Maximum RAM usage: {max_ram_usage:.3f} MB')
    detailed_ram_data = profile_ram(model, input_tensor, num_bytes=opt.
        num_bytes, detailed=True)
    ram_report(detailed_ram_data, verbose=opt.verbose, export=True,
        filename='ram_usage_report')
    return
