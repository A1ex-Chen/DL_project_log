def run(args):
    args = candle.ArgumentStruct(**args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f'cuda' if args.cuda else 'cpu')
    train_loader, valid_loader, test_loader = create_data_loaders(args)
    config = BertConfig(num_attention_heads=2, hidden_size=128,
        num_hidden_layers=1, num_labels=args.num_classes)
    model = BertForSequenceClassification(config)
    model.to(args.device)
    params = [{'params': [p for n, p in model.named_parameters()],
        'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=args.eps)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, criterion, args, epoch)
        validate(valid_loader, model, args, args.device, epoch)
    quantized_model = torch.quantization.quantize_dynamic(model.to('cpu'),
        {torch.nn.Linear}, dtype=torch.qint8)
    model = model.to('cpu')
    if args.verbose:
        print(quantized_model)
    print_size_of_model(model)
    print_size_of_model(quantized_model)
    time_evaluation(valid_loader, model, args, device='cpu')
    time_evaluation(valid_loader, quantized_model, args, device='cpu')
