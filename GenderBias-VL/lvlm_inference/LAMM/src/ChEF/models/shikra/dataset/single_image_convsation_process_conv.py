def process_conv(self, raw_conv: List[Dict[str, Any]]) ->List[Dict[str, Any]]:
    """
        some utils preprocess for raw_conv.
            e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
        """
    return self.process_func['conv'](raw_conv, self.preprocessor, self.
        conv_template)
