def print_results(self, result_dict, type_label):
    self.print_fn(80 * '-')
    self.print_fn('Model Name'.center(30) + 'Batch Size'.center(15) +
        'Seq Length'.center(15) + type_label.center(15))
    self.print_fn(80 * '-')
    for model_name in self.args.model_names:
        for batch_size in result_dict[model_name]['bs']:
            for sequence_length in result_dict[model_name]['ss']:
                result = result_dict[model_name]['result'][batch_size][
                    sequence_length]
                if isinstance(result, float):
                    result = round(1000 * result) / 1000
                    result = '< 0.001' if result == 0.0 else str(result)
                else:
                    result = str(result)
                self.print_fn(model_name[:30].center(30) + str(batch_size).
                    center(15), str(sequence_length).center(15), result.
                    center(15))
    self.print_fn(80 * '-')
