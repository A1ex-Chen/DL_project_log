@torch.no_grad()
def multi_select(self, images, texts, answers, num_cand=None):
    all_losses = []
    for answer in answers:
        choice_samples = {'image': images, 'instruction_input': texts,
            'answer': answer}
        loss = self.forward(choice_samples, reduction='none')['loss'].reshape(
            -1, 1)
        all_losses.append(loss)
        torch.cuda.empty_cache()
    all_losses = torch.cat(all_losses, dim=-1)
    if num_cand is not None:
        for i in range(all_losses.shape[0]):
            all_losses[i, num_cand[i]:] = 9999
    output_class_ranks = torch.argsort(all_losses, dim=-1)
    return output_class_ranks.tolist()
