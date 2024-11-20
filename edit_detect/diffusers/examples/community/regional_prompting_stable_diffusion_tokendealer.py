def tokendealer(self, all_prompts):
    for prompts in all_prompts:
        targets = [p.split(',')[-1] for p in prompts[1:]]
        tt = []
        for target in targets:
            ptokens = self.tokenizer(prompts, max_length=self.tokenizer.
                model_max_length, padding=True, truncation=True,
                return_tensors='pt').input_ids[0]
            ttokens = self.tokenizer(target, max_length=self.tokenizer.
                model_max_length, padding=True, truncation=True,
                return_tensors='pt').input_ids[0]
            tlist = []
            for t in range(ttokens.shape[0] - 2):
                for p in range(ptokens.shape[0]):
                    if ttokens[t + 1] == ptokens[p]:
                        tlist.append(p)
            if tlist != []:
                tt.append(tlist)
    return tt
