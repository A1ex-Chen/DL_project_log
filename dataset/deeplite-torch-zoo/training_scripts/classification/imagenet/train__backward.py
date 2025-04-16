def _backward(_loss):
    if loss_scaler is not None:
        loss_scaler(_loss, optimizer, clip_grad=args.clip_grad, clip_mode=
            args.clip_mode, parameters=model_parameters(model, exclude_head
            ='agc' in args.clip_mode), create_graph=second_order,
            need_update=need_update)
    else:
        _loss.backward(create_graph=second_order)
        if need_update:
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(model_parameters(model,
                    exclude_head='agc' in args.clip_mode), value=args.
                    clip_grad, mode=args.clip_mode)
            optimizer.step()
