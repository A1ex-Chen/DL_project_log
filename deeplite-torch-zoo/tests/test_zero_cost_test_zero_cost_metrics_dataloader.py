@pytest.mark.parametrize(('metric_name', 'ref_value'), REF_METRIC_VALUES)
def test_zero_cost_metrics_dataloader(metric_name, ref_value,
    cifar100_dataloaders, rel_tolerance=0.01):
    init_seeds(42)
    model = get_model(model_name='resnet18', dataset_name='cifar100',
        pretrained=False)
    loss = nn.CrossEntropyLoss()
    estimator_fn = get_zero_cost_estimator(metric_name=metric_name)
    metric_value = estimator_fn(model, dataloader=cifar100_dataloaders[
        'test'], loss_fn=loss)
    assert pytest.approx(metric_value, rel_tolerance) == ref_value
