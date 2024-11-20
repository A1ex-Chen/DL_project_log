def evaluate(eval_iter, model, args):
    model.eval()
    if args.mem_len == 0:
        model.reset_length(tgt_len=args.eval_tgt_len, ext_len=args.ext_len +
            args.tgt_len - args.eval_tgt_len, mem_len=args.mem_len)
    else:
        model.reset_length(tgt_len=args.eval_tgt_len, ext_len=args.ext_len,
            mem_len=args.mem_len + args.tgt_len - args.eval_tgt_len)
    total_len, total_loss = 0, 0.0
    with torch.no_grad():
        mems = None
        for i, (data, target, seq_len, warm) in enumerate(eval_iter):
            if args.eval_max_steps > 0 and i >= args.eval_max_steps:
                break
            loss, mems = model(data, target, mems)
            loss = loss.float().mean()
            if warm:
                total_loss += seq_len * loss.item()
                total_len += seq_len
    model.reset_length(tgt_len=args.tgt_len, ext_len=args.ext_len, mem_len=
        args.mem_len)
    model.train()
    return total_loss / total_len
