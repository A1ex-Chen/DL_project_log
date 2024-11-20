def forward(self, samples, conditions=None):
    if conditions is None:
        return self.__clf_net(self.__encoder(samples))
    else:
        return self.__clf_net(torch.cat((self.__encoder(samples),
            conditions), dim=1))
