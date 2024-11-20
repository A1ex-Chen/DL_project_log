def capture_graph(self):
    list_encode_segment = []
    list_predict_segment = []
    list_max_feat_len = []
    list_max_txt_len = []
    for i in range(self.num_cg):
        list_max_feat_len.append(self.max_feat_len - i * self.max_feat_len //
            self.num_cg)
        list_encode_segment.append(self._gen_encode_graph(list_max_feat_len[i])
            )
        list_max_txt_len.append(self.max_txt_len - i * self.max_txt_len //
            self.num_cg)
        list_predict_segment.append(self._gen_predict_graph(
            list_max_txt_len[i]))
    self.dict_encode_graph = {}
    curr_list_ptr = len(list_max_feat_len) - 1
    for feat_len in range(1, self.max_feat_len + 1):
        while feat_len > list_max_feat_len[curr_list_ptr]:
            curr_list_ptr -= 1
            assert curr_list_ptr >= 0
        self.dict_encode_graph[feat_len] = list_max_feat_len[curr_list_ptr
            ], list_encode_segment[curr_list_ptr]
    self.dict_predict_graph = {}
    curr_list_ptr = len(list_max_txt_len) - 1
    for txt_len in range(1, self.max_txt_len + 1):
        while txt_len > list_max_txt_len[curr_list_ptr]:
            curr_list_ptr -= 1
            assert curr_list_ptr >= 0
        self.dict_predict_graph[txt_len] = list_max_txt_len[curr_list_ptr
            ], list_predict_segment[curr_list_ptr]
