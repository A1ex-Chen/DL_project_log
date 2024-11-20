def build_dataset_for_caption_inference(self, src_tokens, src_lengths,
    img_src_tokens, img_gpt_input_mask, **kwargs):
    """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
    dataset = StripTokenDataset(TokenBlockDataset(src_tokens, src_lengths,
        block_size=None, pad=self.source_dictionary.pad(), eos=self.
        source_dictionary.eos(), break_mode='eos'), self.source_dictionary.
        eos())
    img_gpt_input_mask = StripTokenDataset(TokenBlockDataset(
        img_gpt_input_mask, src_lengths, block_size=None, pad=self.
        source_dictionary.pad(), eos=self.source_dictionary.eos(),
        break_mode='eos'), self.source_dictionary.eos())
    src_dataset = dataset
    tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.
        pad())
    return NestedDictionaryDataset({'id': IdDataset(), 'net_input': {
        'src_tokens': PadDataset(src_dataset, pad_idx=self.
        source_dictionary.pad(), left_pad=True), 'img_src_tokens':
        RawImageDataset(img_src_tokens), 'img_gpt_input_mask': PadDataset(
        img_gpt_input_mask, pad_idx=0, left_pad=True), 'src_lengths':
        NumelDataset(src_dataset, reduce=False)}, 'target': PadDataset(
        tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=True)},
        sizes=[np.array(src_lengths)])
