def compile_model(model, device, args):
    inp = torch.randint(0, 1000, (args.tgt_len, args.batch_size)).to(device)
    tgt = torch.randint(0, 1000, (args.tgt_len, args.batch_size)).to(device)
    start = time.time()
    with torch.no_grad():
        mems = None
        for _ in range(2):
            _, mems = model(inp, tgt, mems)
    torch.cuda.synchronize()
    stop = time.time()
    logging.info(f'Building the model took {stop - start:.2f} seconds')
