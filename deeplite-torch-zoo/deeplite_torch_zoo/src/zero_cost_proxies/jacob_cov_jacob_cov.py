@ZERO_COST_SCORES.register('jacob_cov')
def jacob_cov(model, model_output_generator, loss_fn=None,
    output_post_processing=None, reduction='sum'):
    if output_post_processing is None:
        output_post_processing = lambda tensors: torch.cat([x.flatten() for
            x in tensors])
    jacobs, _ = get_jacob(model, model_output_generator, output_post_processing
        )
    try:
        jc = eval_score(jacobs.reshape(jacobs.size(0), -1).cpu().numpy())
    except:
        jc = np.nan
    return aggregate_statistic(jc, reduction=reduction)
