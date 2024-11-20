def ddp_computation(self):
    resp = pm.DDPBucketSizesComputationTimes()
    try:
        analysis_results = ddp_analysis(self._model_provider, self.
            _input_provider, self._iteration_provider)
        resp.forward_time_ms = float(analysis_results['forward_time_ms'])
        resp.bucket_sizes.extend(analysis_results['bucket_sizes'])
        for computation_time_item in analysis_results[
            'expected_computation_times']:
            add_item_to_resp = resp.computation_times.add()
            add_item_to_resp.ngpus = int(computation_time_item['ngpus'])
            add_item_to_resp.expected_max_times.extend(computation_time_item
                ['expected_max_times'])
    except AnalysisError as ex:
        message = str(ex)
        logger.error(message)
        resp.analysis_error.error_message = message
    except Exception as ex:
        message = str(ex)
        logger.error(message)
        logger.error('There was an error measuring ddp throughput')
        resp.analysis_error.error_message = (
            'There was an error measuring ddp throughput')
    finally:
        return resp
