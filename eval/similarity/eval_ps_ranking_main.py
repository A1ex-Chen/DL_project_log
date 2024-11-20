def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_acc_dict = {k: [] for k in MODEL_PATH.keys()}
    for model_name, model_path in MODEL_PATH.items():
        model_name_acc_dict[model_name] = {'train': [], 'test': []}
        results = {}
        phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.
            full_run_mode, args.task, 'train', model_path, device, shuffle=
            True, contextual=args.contextual)
        phrase1_tensor = phrase1_tensor.cpu().detach().numpy()
        phrase2_tensor = phrase2_tensor.cpu().detach().numpy()
        label_tensor = label_tensor.cpu().detach().numpy()
        (best_threshold, best_acc, best_acc_pos, avg_cos_sim, std_cos_sim) = (
            train_or_evaluate(phrase1_tensor, phrase2_tensor, label_tensor))
        model_name_acc_dict[model_name]['train'] = [best_threshold,
            best_acc, best_acc_pos, avg_cos_sim, std_cos_sim]
        results['train'] = {'best_threshold': '{:.2f}'.format(
            best_threshold * 100), 'best_acc': '{:.2f}'.format(best_acc * 
            100), 'best_acc_pos': '{:.2f}'.format(best_acc_pos * 100),
            'avg_cos_sim': '{:.2f}'.format(avg_cos_sim * 100),
            'std_cos_sim': '{:.2f}'.format(std_cos_sim * 100)}
        phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.
            full_run_mode, args.task, 'test', model_path, device, shuffle=
            False, contextual=args.contextual)
        phrase1_tensor = phrase1_tensor.cpu().detach().numpy()
        phrase2_tensor = phrase2_tensor.cpu().detach().numpy()
        label_tensor = label_tensor.cpu().detach().numpy()
        (best_threshold, best_acc, best_acc_pos, avg_cos_sim, std_cos_sim) = (
            train_or_evaluate(phrase1_tensor, phrase2_tensor, label_tensor,
            best_threshold=best_threshold))
        model_name_acc_dict[model_name]['test'] = [best_threshold, best_acc,
            best_acc_pos, avg_cos_sim, std_cos_sim]
        results['test'] = {'best_threshold': '{:.2f}'.format(best_threshold *
            100), 'best_acc': '{:.2f}'.format(best_acc * 100),
            'best_acc_pos': '{:.2f}'.format(best_acc_pos * 100),
            'avg_cos_sim': '{:.2f}'.format(avg_cos_sim * 100),
            'std_cos_sim': '{:.2f}'.format(std_cos_sim * 100)}
        result_dir = args.result_dir + ('contextual/' if args.contextual else
            'non_contextual/')
        if not exists(result_dir):
            makedirs(result_dir)
        output_fname = os.path.join(result_dir,
            f'{args.task}_{model_name}.json')
        with open(output_fname, 'w') as outfile:
            json.dump(results, outfile, indent=4)
        print(model_name_acc_dict)
        print(f'\n finished {model_name}\n')
    for k, v in model_name_acc_dict.items():
        print(
            f"model: {k} \tbest threshold: \t{v['train'][0]:.4f} \tbest accuracy: \t{v['train'][1]:.4f} \tbest accuracy (pos): \t{v['train'][2]:.4f} \taverage cosine score: \t{v['train'][3]:.4f} \tstd cosine score: \t{v['train'][4]:.4f}"
            )
    for k, v in model_name_acc_dict.items():
        print(
            f"model: {k} \tbest threshold: \t{v['test'][0]:.4f} \tbest accuracy: \t{v['test'][1]:.4f} \tbest accuracy (pos): \t{v['test'][2]:.4f} \taverage cosine score: \t{v['test'][3]:.4f} \tstd cosine score: \t{v['test'][4]:.4f}"
            )
    print('Done with main --- {}'.format(args.task))
