def forward(self, c_concat, c_crossattn):
    c_concat = self.concat_conditioner(c_concat)
    c_crossattn = self.crossattn_conditioner(c_crossattn)
    return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}
