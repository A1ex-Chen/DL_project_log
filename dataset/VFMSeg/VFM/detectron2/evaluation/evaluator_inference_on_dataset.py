def inference_on_dataset(model, data_loader, evaluator: Union[
    DatasetEvaluator, List[DatasetEvaluator], None]):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info('Start inference on {} batches'.format(len(data_loader)))
    total = len(data_loader)
    if evaluator is None:
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time
                ) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter *
                    (total - idx - 1)))
                log_every_n_seconds(logging.INFO,
                    f'Inference done {idx + 1}/{total}. Dataloading: {data_seconds_per_iter:.4f} s/iter. Inference: {compute_seconds_per_iter:.4f} s/iter. Eval: {eval_seconds_per_iter:.4f} s/iter. Total: {total_seconds_per_iter:.4f} s/iter. ETA={eta}'
                    , n=5)
            start_data_time = time.perf_counter()
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        'Total inference time: {} ({:.6f} s / iter per device, on {} devices)'
        .format(total_time_str, total_time / (total - num_warmup), num_devices)
        )
    total_compute_time_str = str(datetime.timedelta(seconds=int(
        total_compute_time)))
    logger.info(
        'Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)'
        .format(total_compute_time_str, total_compute_time / (total -
        num_warmup), num_devices))
    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results
