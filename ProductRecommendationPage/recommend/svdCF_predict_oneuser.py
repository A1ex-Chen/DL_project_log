def predict_oneuser(self, uid, iid):
    self.bi.setdefault(iid, 0)
    self.bu.setdefault(uid, 0)
    self.qi.setdefault(iid, np.zeros((self.K, 1)))
    self.pu.setdefault(uid, np.zeros((self.K, 1)))
    rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] *
        self.pu[uid])
    if rating > 3:
        rating = 3
    if rating < 1:
        rating = 1
    return rating
