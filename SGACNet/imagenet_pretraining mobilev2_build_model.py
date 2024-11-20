def build_model(args):


    class Classifier(nn.Module):

        def __init__(self):
            super().__init__()
            if args.encoder == 'mobilenet_v2':
                Encoder = mobilenet_v2
            elif args.encoder == 'mobilenet_v2':
                Encoder = mobilenet_v2
            self.encoder = Encoder(block='NonBottleneck1D',
                pretrained_on_imagenet=False, activation=nn.ReLU(inplace=True))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def forward(self, images):
            encoder_outs = self.encoder(images)
            enc_down_32, enc_down_16, enc_down_8, enc_down_4 = encoder_outs
            out = self.avgpool(enc_down_32)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out
    model = Classifier()
    print(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    model.to(device)
    return model, device
