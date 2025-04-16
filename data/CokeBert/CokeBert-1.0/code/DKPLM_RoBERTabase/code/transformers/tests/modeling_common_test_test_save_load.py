def test_save_load(self):
    config, inputs_dict = (self.model_tester.
        prepare_config_and_inputs_for_common())
    for model_class in self.all_model_classes:
        model = model_class(config)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs_dict)
        with TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = model_class.from_pretrained(tmpdirname)
            with torch.no_grad():
                after_outputs = model(**inputs_dict)
            out_1 = after_outputs[0].numpy()
            out_2 = outputs[0].numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-05)
