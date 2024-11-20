def collate_active_info(src_seq, src_enc, inst_idx_to_position_map,
    active_inst_idx_list):
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in
        active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)
    active_src_seq = collect_active_part(src_seq, active_inst_idx,
        n_prev_active_inst, n_bm)
    active_src_enc = collect_active_part(src_enc, active_inst_idx,
        n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
        active_inst_idx_list)
    return active_src_seq, active_src_enc, active_inst_idx_to_position_map
