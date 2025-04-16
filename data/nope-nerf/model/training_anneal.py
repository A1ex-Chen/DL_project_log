def anneal(self, start_weight, end_weight, anneal_start_epoch,
    anneal_epoches, current):
    """Anneal the weight from start_weight to end_weight
        """
    if current <= anneal_start_epoch:
        return start_weight
    elif current >= anneal_start_epoch + anneal_epoches:
        return end_weight
    else:
        return start_weight + (end_weight - start_weight) * (current -
            anneal_start_epoch) / anneal_epoches
