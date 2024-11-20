def batched_generate(generator, examples, **kwargs):
    preds_list = []
    with torch.no_grad():
        for e in range(len(examples)):
            preds_list += generator(examples[e:e + 1], early_stopping=True,
                **kwargs)
    return preds_list
