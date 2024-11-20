def habitat_predict(self):
    resp = pm.HabitatResponse()
    if not habitat_found:
        logger.debug('Skipping deepview predictions, returning empty response.'
            )
        return resp
    try:
        print('deepview_predict: begin')
        hc.release_cupti_hook()
        DEVICES = [habitat.Device.P100, habitat.Device.P4000, habitat.
            Device.RTX2070, habitat.Device.RTX2080Ti, habitat.Device.T4,
            habitat.Device.V100, habitat.Device.A100, habitat.Device.
            RTX3090, habitat.Device.A40, habitat.Device.A4000, habitat.
            Device.RTX4000, habitat.Device.L4, habitat.Device.H100]
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            raise Exception(
                'NVML failed to find a GPU. Please ensure that you                 have a NVIDIA GPU installed and that the drivers are functioning                 correctly.'
                )
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        source_device_name = pynvml.nvmlDeviceGetName(nvml_handle).decode(
            'utf-8')
        split_source_device_name = re.split('-|\\s|_|\\\\|/',
            source_device_name)
        source_device = (None if logging.root.level > logging.DEBUG else
            habitat.Device.T4)
        for device in DEVICES:
            if device.name in ''.join(split_source_device_name):
                source_device = device
        pynvml.nvmlShutdown()
        if not source_device:
            logger.debug(
                'Skipping DeepView predictions,                    source not in list of supported GPUs.'
                )
            src = pm.HabitatDevicePrediction()
            src.device_name = 'unavailable'
            src.runtime_ms = -1
            resp.predictions.append(src)
            return resp
        print('deepview_predict: detected source device', source_device.name)
        model = self._model_provider()
        inputs = self._input_provider()
        iteration = self._iteration_provider(model)

        def runnable():
            iteration(*inputs)
        profiler = RunTimeProfiler()
        context = Context(origin_device=source_device, profiler=profiler,
            percentile=99.5)
        threshold = self.habitat_compute_threshold(runnable, context)
        tracker = habitat.OperationTracker(device=context.origin_device,
            metrics=[habitat.Metric.SinglePrecisionFLOPEfficiency, habitat.
            Metric.DRAMReadBytes, habitat.Metric.DRAMWriteBytes],
            metrics_threshold_ms=threshold)
        with tracker.track():
            iteration(*inputs)
        print('deepview_predict: tracing on origin device')
        trace = tracker.get_tracked_trace()
        src = pm.HabitatDevicePrediction()
        src.device_name = 'source'
        src.runtime_ms = trace.run_time_ms
        resp.predictions.append(src)
        for device in DEVICES:
            print('deepview_predict: predicting for', device)
            predicted_trace = trace.to_device(device)
            pred = pm.HabitatDevicePrediction()
            pred.device_name = device.name
            pred.runtime_ms = predicted_trace.run_time_ms
            resp.predictions.append(pred)
        print(f'returning {len(resp.predictions)} predictions.')
        hc.release_cupti_hook()
    except AnalysisError as ex:
        message = str(ex)
        logger.error(message)
        resp.analysis_error.error_message = message
    except Exception as ex:
        message = str(ex)
        logger.error('There was an error running DeepView Predict')
        logger.error(message)
        cupti_context_error = ''
        if 'CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED' in message:
            cupti_context_error = """We have detected that cuda versions <= 11.8 maybe not close CUPTI context properly between each execution.
                Please try with a version of pytorch that includes cuda version >= 12.1. Or you may close this window and open a new one."""
            print(cupti_context_error)
        resp.analysis_error.error_message = (
            'There was an error running DeepView Predict\n' +
            cupti_context_error)
    finally:
        return resp
