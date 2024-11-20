def compare_models(self, modelA, modelB, test_setup=''):
    state_dictA = modelA.state_dict()
    state_dictB = modelB.state_dict()
    self.assertEqual(len(state_dictA), len(state_dictB), 
        'state_dicts have different lengths' + test_setup)
    for key in state_dictA:
        paramA = state_dictA[key]
        paramB = state_dictB[key]
        self.assertTrue((paramA == paramB).all(), msg=
            'Parameters in state_dices not equal.' +
            """key: {}
param: {}
restored: {}
diff: {} for {}""".format(key,
            paramA, paramB, paramA - paramB, test_setup))
