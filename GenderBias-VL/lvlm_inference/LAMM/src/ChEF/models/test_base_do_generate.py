@torch.no_grad()
def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
    """
            Direct generate answers with images and questions, max_len(answer) = max_new_tokens
        """
    raise NotImplementedError
