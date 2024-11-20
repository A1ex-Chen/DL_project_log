def zero_shot_eval(model, data, epoch, args):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if epoch % args.zeroshot_frequency != 0 and epoch != args.epochs:
        return {}
    logging.info('Starting zero-shot imagenet.')
    logging.info('Building zero-shot classifier')
    classifier = zero_shot_classifier(model, imagenet_classnames,
        openai_imagenet_template, args)
    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader,
            args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader,
            args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    logging.info('Finished zero-shot imagenet.')
    return results
