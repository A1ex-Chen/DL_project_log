@pytest.mark.parametrize(('metric_name', 'ref_value'), REF_METRIC_VALUES)
def test_zero_cost_metrics_generator(metric_name, ref_value,
    cifar100_dataloaders, rel_tolerance=0.01):
    init_seeds(42)
    model = get_model(model_name='resnet18', dataset_name='cifar100',
        pretrained=False)

    def data_generator(model, shuffle_data=True, input_gradient=False):
        test_loader = cifar100_dataloaders['test'] if shuffle_data else repeat(
            next(iter(cifar100_dataloaders['test'])))
        for inputs, targets in test_loader:
            inputs.requires_grad_(input_gradient)
            outputs = model(inputs)
            yield inputs, outputs, targets, {'loss_kwarg': None}
    loss = nn.CrossEntropyLoss()

    def loss_fn(outputs, targets, **kwargs):
        assert kwargs['loss_kwarg'] is None
        return loss(outputs, targets)
    estimator_fn = get_zero_cost_estimator(metric_name=metric_name)
    metric_value = estimator_fn(model, model_output_generator=
        data_generator, loss_fn=loss_fn)
    assert pytest.approx(metric_value, rel=rel_tolerance) == ref_value
