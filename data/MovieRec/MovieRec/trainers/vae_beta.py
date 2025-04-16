@property
def beta(self):
    if self.model.training:
        self.__beta = min(self.__beta + self.anneal_amount, self.anneal_cap)
    return self.__beta
