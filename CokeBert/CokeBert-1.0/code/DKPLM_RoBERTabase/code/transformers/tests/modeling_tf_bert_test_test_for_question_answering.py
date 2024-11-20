def test_for_question_answering(self):
    config_and_inputs = self.model_tester.prepare_config_and_inputs()
    self.model_tester.create_and_check_bert_for_question_answering(*
        config_and_inputs)
