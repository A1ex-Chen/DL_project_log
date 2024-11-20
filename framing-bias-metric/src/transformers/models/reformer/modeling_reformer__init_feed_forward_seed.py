def _init_feed_forward_seed(self):
    """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
    if hasattr(torch.cuda, 'default_generators') and len(torch.cuda.
        default_generators) > 0:
        device_idx = torch.cuda.current_device()
        self.feed_forward_seed = torch.cuda.default_generators[device_idx
            ].seed()
    else:
        self.feed_forward_seed = int(torch.seed() % sys.maxsize)
    torch.manual_seed(self.feed_forward_seed)
