def process_evaluation_epoch(aggregates):
    """
    Processes results from each worker at the end of evaluation and combine to final result
    Args:
        aggregates: dictionary containing information of entire evaluation
    Return:
        wer: final word error rate
        loss: final loss
    """
    if 'losses' in aggregates:
        eloss = torch.mean(torch.stack(aggregates['losses'])).item()
    else:
        eloss = None
    hypotheses = aggregates['preds']
    references = aggregates['txts']
    wer, scores, num_words = word_error_rate(hypotheses, references)
    multi_gpu = dist.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= dist.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()
        scores_tensor = torch.tensor(scores).cuda()
        dist.all_reduce(scores_tensor)
        scores = scores_tensor.item()
        num_words_tensor = torch.tensor(num_words).cuda()
        dist.all_reduce(num_words_tensor)
        num_words = num_words_tensor.item()
        wer = scores * 1.0 / num_words
    return wer, eloss
