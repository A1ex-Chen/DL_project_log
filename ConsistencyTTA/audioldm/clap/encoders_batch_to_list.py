def batch_to_list(self, batch):
    ret = []
    for i in range(batch.size(0)):
        ret.append(batch[i])
    return ret
