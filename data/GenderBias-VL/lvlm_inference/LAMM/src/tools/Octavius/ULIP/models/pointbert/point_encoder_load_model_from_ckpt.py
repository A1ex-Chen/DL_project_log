def load_model_from_ckpt(self, bert_ckpt_path):
    ckpt = torch.load(bert_ckpt_path)
    base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['base_model']
        .items()}
    for k in list(base_ckpt.keys()):
        if k.startswith('transformer_q') and not k.startswith(
            'transformer_q.cls_head'):
            base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
        elif k.startswith('base_model'):
            base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
        del base_ckpt[k]
    incompatible = self.load_state_dict(base_ckpt, strict=False)
    if incompatible.missing_keys:
        print_log('missing_keys', logger='Transformer')
        print_log(get_missing_parameters_message(incompatible.missing_keys),
            logger='Transformer')
    if incompatible.unexpected_keys:
        print_log('unexpected_keys', logger='Transformer')
        print_log(get_unexpected_parameters_message(incompatible.
            unexpected_keys), logger='Transformer')
    print_log(
        f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}',
        logger='Transformer')
