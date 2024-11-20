def main(model_path, output_path):
    unet_state_dict = load_file(model_path)
    config = {}
    converted_state_dict = convert_state_dict(unet_state_dict)
    unet = Kandinsky3UNet(config)
    unet.load_state_dict(converted_state_dict)
    unet.save_pretrained(output_path)
    print(f'Converted model saved to {output_path}')
