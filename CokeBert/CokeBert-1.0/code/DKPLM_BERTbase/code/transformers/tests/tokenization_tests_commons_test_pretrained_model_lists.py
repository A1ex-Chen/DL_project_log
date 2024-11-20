def test_pretrained_model_lists(self):
    weights_list = list(self.tokenizer_class.max_model_input_sizes.keys())
    weights_lists_2 = []
    for file_id, map_list in self.tokenizer_class.pretrained_vocab_files_map.items(
        ):
        weights_lists_2.append(list(map_list.keys()))
    for weights_list_2 in weights_lists_2:
        self.assertListEqual(weights_list, weights_list_2)
