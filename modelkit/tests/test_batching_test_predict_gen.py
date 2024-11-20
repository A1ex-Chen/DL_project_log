def test_predict_gen():


    class GeneratorTestModel(Model):

        def _predict_batch(self, items):
            return [(item, position_in_batch, len(items)) for 
                position_in_batch, item in enumerate(items)]
    m = GeneratorTestModel()
    _do_gen_test(m, 16, 66)
    _do_gen_test(m, 10, 8)
    _do_gen_test(m, 100, 5)
