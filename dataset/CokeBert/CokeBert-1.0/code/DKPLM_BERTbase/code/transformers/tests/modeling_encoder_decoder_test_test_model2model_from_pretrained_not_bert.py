def test_model2model_from_pretrained_not_bert(self):
    logging.basicConfig(level=logging.INFO)
    with self.assertRaises(ValueError):
        _ = Model2Model.from_pretrained('roberta')
    with self.assertRaises(ValueError):
        _ = Model2Model.from_pretrained('distilbert')
    with self.assertRaises(ValueError):
        _ = Model2Model.from_pretrained('does-not-exist')
