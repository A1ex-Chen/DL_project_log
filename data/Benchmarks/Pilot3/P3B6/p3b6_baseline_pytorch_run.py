def run(args):
    args = candle.ArgumentStruct(**args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f'cuda' if args.cuda else 'cpu')
    train_loader, valid_loader, test_loader = create_data_loaders(args)
    config = BertConfig(num_attention_heads=2, hidden_size=128,
        num_hidden_layers=1, num_labels=args.num_classes)
    model = BertForSequenceClassification(config)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
        eps=args.eps)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, criterion, args, epoch)
        validate(valid_loader, model, args, epoch)
