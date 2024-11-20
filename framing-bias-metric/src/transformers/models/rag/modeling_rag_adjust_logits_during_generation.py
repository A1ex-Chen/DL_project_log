def adjust_logits_during_generation(self, logits, cur_len, max_length):
    return self.rag.generator.adjust_logits_during_generation(logits,
        cur_len=cur_len, max_length=max_length)
