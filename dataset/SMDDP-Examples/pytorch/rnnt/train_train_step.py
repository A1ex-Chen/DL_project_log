def train_step(model, loss_fn, args, batch_size, feats, feat_lens, txt,
    txt_lens, optimizer, grad_scaler, meta_data, train_loader, rnnt_graph,
    copy_stream, pred_stream):
    lr_cpu = torch.tensor(0, dtype=torch.float, device='cpu').pin_memory()
    loss_cpu = torch.tensor(0, dtype=torch.float16, device='cpu').pin_memory()
    if args.batch_split_factor == 1:
        if rnnt_graph is not None:
            log_probs, log_prob_lens = rnnt_graph.step(feats, feat_lens,
                txt, txt_lens, meta_data[0])
        else:
            log_probs, log_prob_lens = model(feats, feat_lens, txt,
                txt_lens, meta_data[0])
        loss = loss_fn(log_probs, log_prob_lens, txt, txt_lens, meta_data[0])
        if args.enable_prefetch and train_loader is not None:
            train_loader.data_iterator().prefetch()
        loss /= args.grad_accumulation_steps
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            loss_cpu.copy_(loss.detach(), non_blocking=True)
            if args.dist_lamb:
                lr_cpu.copy_(optimizer._lr, non_blocking=True)
        del log_probs, log_prob_lens
        if args.dist_lamb:
            grad_scaler.scale(loss).backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        copy_stream.synchronize()
        if torch.isnan(loss_cpu).any():
            raise Exception('Loss is NaN')
        return loss_cpu.item(), lr_cpu.item()
    else:
        f, g, log_prob_lens = model.enc_pred(feats, feat_lens, txt,
            txt_lens, pred_stream)
        f_2, g_2 = f.detach(), g.detach()
        f_2.requires_grad = True
        g_2.requires_grad = True
        B_split = batch_size // args.batch_split_factor
        loss_item = 0
        for i in range(args.batch_split_factor):
            log_probs = model.joint(f_2[i * B_split:(i + 1) * B_split], g_2
                [i * B_split:(i + 1) * B_split], args.apex_transducer_joint,
                log_prob_lens[i * B_split:(i + 1) * B_split], meta_data[i])
            loss = loss_fn(log_probs, log_prob_lens[i * B_split:(i + 1) *
                B_split], txt[i * B_split:(i + 1) * B_split], txt_lens[i *
                B_split:(i + 1) * B_split], meta_data[i])
            if args.enable_prefetch and train_loader is not None and i == 0:
                train_loader.data_iterator().prefetch()
            loss /= args.grad_accumulation_steps * args.batch_split_factor
            copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                loss_cpu.copy_(loss.detach(), non_blocking=True)
                if args.dist_lamb and i == 0:
                    lr_cpu.copy_(optimizer._lr, non_blocking=True)
            del log_probs
            if args.dist_lamb:
                grad_scaler.scale(loss).backward()
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            copy_stream.synchronize()
            if torch.isnan(loss_cpu).any():
                raise Exception('Loss is NaN')
            loss_item += loss_cpu.item()
        f.backward(f_2.grad)
        g.backward(g_2.grad)
        return loss_item, lr_cpu.item()
