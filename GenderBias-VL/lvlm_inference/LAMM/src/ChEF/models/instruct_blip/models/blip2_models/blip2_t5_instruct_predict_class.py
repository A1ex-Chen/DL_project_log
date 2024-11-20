def predict_class(self, samples, candidates, n_segments=1):
    if type(candidates[0]) == list:
        results = []
        for i in range(samples['image'].size(0)):
            this_sample = {'image': samples['image'][i].unsqueeze(0),
                'prompt': samples['prompt']}
            if 'text_input' in samples.keys():
                this_sample['text_input'] = [samples['text_input'][i]]
            if 'context' in samples.keys():
                this_sample['context'] = [samples['context'][i]]
            if 'history' in samples.keys():
                this_sample['history'] = [samples['history'][i]]
            if 'caption' in samples.keys():
                this_sample['caption'] = [samples['caption'][i]]
            this_result = self._predict_class(this_sample, candidates[i],
                n_segments)
            results.append(this_result)
        try:
            results = torch.cat(results, dim=0)
        except:
            results = [res.tolist()[0] for res in results]
        return results
    return self._predict_class(samples, candidates, n_segments)
