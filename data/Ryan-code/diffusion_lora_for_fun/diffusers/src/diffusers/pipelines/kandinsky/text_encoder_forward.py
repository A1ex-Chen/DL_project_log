def forward(self, input_ids, attention_mask):
    embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask
        )[0]
    embs2 = (embs * attention_mask.unsqueeze(2)).sum(dim=1
        ) / attention_mask.sum(dim=1)[:, None]
    return self.LinearTransformation(embs2), embs
