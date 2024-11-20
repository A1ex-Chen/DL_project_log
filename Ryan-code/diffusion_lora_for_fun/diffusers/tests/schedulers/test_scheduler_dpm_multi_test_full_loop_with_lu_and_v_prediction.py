def test_full_loop_with_lu_and_v_prediction(self):
    sample = self.full_loop(prediction_type='v_prediction', use_lu_lambdas=True
        )
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.1554) < 0.001
