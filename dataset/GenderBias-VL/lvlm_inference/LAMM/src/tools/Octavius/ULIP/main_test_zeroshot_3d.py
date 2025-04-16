def test_zeroshot_3d(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    tokenizer = SimpleTokenizer()
    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args
        .batch_size, shuffle=False, num_workers=args.workers, pin_memory=
        True, sampler=None, drop_last=False)
    results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)
    return results
