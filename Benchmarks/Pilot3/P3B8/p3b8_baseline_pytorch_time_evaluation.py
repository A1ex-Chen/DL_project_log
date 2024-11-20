def time_evaluation(dataloader, model, args, device):
    s = time.time()
    loss = validate(dataloader, model, args, device, epoch=0)
    elapsed = time.time() - s
    print(f'\telapsed time (seconds): {elapsed:.1f}')
