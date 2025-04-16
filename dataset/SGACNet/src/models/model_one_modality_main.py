def main():
    """
    Useful to check if model is built correctly.
    """
    model = SGACNetOneModality()
    print(model)
    model.eval()
    rgb_image = torch.randn(1, 3, 1080, 1920)
    from torch.autograd import Variable
    inputs_rgb = Variable(rgb_image)
    with torch.no_grad():
        output = model(inputs_rgb)
    print(output.shape)
