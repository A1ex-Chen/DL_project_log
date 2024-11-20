def make_file_list(self, output_files, json_names):
    file_name = hash_list_of_strings(json_names)
    if self.dist_sampler:
        file_name += '__%d' % self.rank
    self.file_list_path = os.path.join('/tmp', 'rnnt_dali.file_list.' +
        file_name)
    self.write_file_list(*self.process_output_files(output_files))
