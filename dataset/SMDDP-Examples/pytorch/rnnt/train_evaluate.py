@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, detokenize, ema_model,
    loss_fn, greedy_decoder, amp_level):
    logging.log_start(logging.constants.EVAL_START, metadata=dict(epoch_num
        =epoch))
    start_time = time.time()
    agg = {'preds': [], 'txts': [], 'idx': []}
    greedy_decoder.update_ema_model_eval(ema_model)
    for i, batch in enumerate(val_loader):
        audio, audio_lens, txt, txt_lens = batch
        feats, feat_lens = val_feat_proc([audio, audio_lens])
        if amp_level == 2:
            feats = feats.half()
        pred = greedy_decoder.decode(feats, feat_lens)
        agg['preds'] += helpers.gather_predictions([pred], detokenize)
        agg['txts'] += helpers.gather_transcripts([txt.cpu()], [txt_lens.
            cpu()], detokenize)
    wer, loss = process_evaluation_epoch(agg)
    logging.log_event(logging.constants.EVAL_ACCURACY, value=wer, metadata=
        dict(epoch_num=epoch))
    logging.log_end(logging.constants.EVAL_STOP, metadata=dict(epoch_num=epoch)
        )
    log((epoch,), step, 'dev_ema', {'wer': 100.0 * wer, 'took': time.time() -
        start_time})
    return wer
