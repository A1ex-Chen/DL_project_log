def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor,
    **kwargs) ->bool:
    """call function of stop creteria

        :param torch.LongTensor output_ids: output token ids
        :return bool: stop or not
        """
    flag = 1
    for id, output_id in enumerate(output_ids):
        if self.stop_flag[id] == 1:
            continue
        if self.check_stop(output_id):
            self.stop_flag[id] = 1
        else:
            flag = 0
    if flag == 1:
        return True
    return False
