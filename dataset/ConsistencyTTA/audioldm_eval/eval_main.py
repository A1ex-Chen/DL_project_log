def main(self, dataset_json_path, generated_files_path, groundtruth_path,
    mel_path=None, target_length=1000, limit_num=None):
    self.file_init_check(generated_files_path)
    self.file_init_check(groundtruth_path)
    same_name = self.get_filename_intersection_ratio(generated_files_path,
        groundtruth_path, limit_num=limit_num)
    return self.calculate_metrics(dataset_json_path, generated_files_path,
        groundtruth_path, mel_path, same_name, target_length, limit_num)
