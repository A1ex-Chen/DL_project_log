def main():
    height = 480
    width = 640
    model = SGACNet(height=height, width=width)
    print(model)
    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    depth_image = torch.randn(1, 1, height, width)
    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output.shape)
