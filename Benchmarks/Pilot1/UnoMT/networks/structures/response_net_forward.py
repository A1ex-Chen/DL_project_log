def forward(self, rnaseq, drug_feature, concentration):
    return self.__resp_net(torch.cat((self.__gene_encoder(rnaseq), self.
        __drug_encoder(drug_feature), concentration), dim=1))
