def collect_active_inst_idx_list(inst_beams, word_prob,
    inst_idx_to_position_map):
    active_inst_idx_list = []
    for inst_idx, inst_position in inst_idx_to_position_map.items():
        is_inst_complete = inst_beams[inst_idx].advance(word_prob[
            inst_position])
        if not is_inst_complete:
            active_inst_idx_list += [inst_idx]
    return active_inst_idx_list
