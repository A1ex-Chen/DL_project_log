def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    print('=> encoding captions')
    with open(os.path.join('./data', 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]
    with open(os.path.join('./data', 'labels.json')) as f:
        labels = json.load(f)[args.validate_dataset_name]
    clip_encoder, visual_preprocess = clip.load('ViT-L/14', device='cuda')
    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = clip_encoder.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim
                =-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim
                =-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)
        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)
        for i, (pc, target, target_name) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1
            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            pc_features = utils.get_model(model).encode_pc(pc)[0]
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
            logits_per_pc = pc_features.half() @ text_features.t()
            (acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5)
                )
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1
            if i % args.print_freq == 0:
                progress.display(i)
        top1_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name
                ] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name
                ] / per_class_stats[name]
        top1_accuracy_per_class = collections.OrderedDict(
            top1_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(
            top5_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.
            values()]))
        print(','.join([str(value) for value in top5_accuracy_per_class.
            values()]))
    progress.synchronize()
    print(f'0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc5': top5.avg}
