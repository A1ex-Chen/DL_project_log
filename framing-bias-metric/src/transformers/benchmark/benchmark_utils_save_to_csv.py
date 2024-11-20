def save_to_csv(self, result_dict, filename):
    if not self.args.save_to_csv:
        return
    self.print_fn('Saving results to csv.')
    with open(filename, mode='w') as csv_file:
        assert len(self.args.model_names
            ) > 0, 'At least 1 model should be defined, but got {}'.format(self
            .model_names)
        fieldnames = ['model', 'batch_size', 'sequence_length']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames + ['result'])
        writer.writeheader()
        for model_name in self.args.model_names:
            result_dict_model = result_dict[model_name]['result']
            for bs in result_dict_model:
                for ss in result_dict_model[bs]:
                    result_model = result_dict_model[bs][ss]
                    writer.writerow({'model': model_name, 'batch_size': bs,
                        'sequence_length': ss, 'result': ('{}' if not
                        isinstance(result_model, float) else '{:.4f}').
                        format(result_model)})
