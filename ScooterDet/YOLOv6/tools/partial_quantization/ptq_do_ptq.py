def do_ptq(model, train_loader, batch_number, device):
    model_ptq = quant_model_init(model, device)
    with torch.no_grad():
        collect_stats(model_ptq, train_loader, batch_number, device)
        compute_amax(model_ptq, method='entropy')
    return model_ptq
