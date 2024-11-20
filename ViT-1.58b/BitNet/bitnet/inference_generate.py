def generate(self, input_str, length):
    """Generates a sequence of tokens based on the input string."""
    inp = torch.from_numpy(np.fromstring(input_str, dtype=np.uint8)).long().to(
        self.device)
    sample = self.model.generate(inp[None, ...], length)
    output_str = self.decode_tokens(sample[0])
    return output_str
