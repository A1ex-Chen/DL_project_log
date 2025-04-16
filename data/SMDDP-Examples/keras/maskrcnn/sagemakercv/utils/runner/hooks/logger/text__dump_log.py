def _dump_log(self, log_dict, runner):
    json_log = OrderedDict()
    for k, v in log_dict.items():
        json_log[k] = self._round_float(v)
    if runner.rank == 0:
        with open(self.json_log_path, 'a+') as f:
            dump(json_log, f, file_format='json')
            f.write('\n')
