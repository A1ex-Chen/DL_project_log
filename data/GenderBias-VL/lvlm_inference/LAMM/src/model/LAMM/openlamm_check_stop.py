def check_stop(self, input_ids):
    """check whether to stop generation

        :param list input_ids: input token ids
        :return bool: stop or not
        """
    for stop in self.stops:
        if torch.all(stop == input_ids[-len(stop):]).item():
            return True
    return False
