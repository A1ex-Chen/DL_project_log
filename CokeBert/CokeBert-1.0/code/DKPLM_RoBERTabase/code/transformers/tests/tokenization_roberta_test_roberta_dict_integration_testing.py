def roberta_dict_integration_testing(self):
    tokenizer = self.get_tokenizer()
    self.assertListEqual(tokenizer.encode('Hello world!',
        add_special_tokens=False), [0, 31414, 232, 328, 2])
    self.assertListEqual(tokenizer.encode('Hello world! cécé herlolip 418',
        add_special_tokens=False), [0, 31414, 232, 328, 740, 1140, 12695, 
        69, 46078, 1588, 2])
