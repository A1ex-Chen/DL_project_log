def measure_utilization(self):
    resp = pm.UtilizationResponse()
    try:
        analysis_results = utilization_analysis(self._model_provider, self.
            _input_provider, self._iteration_provider)
        serialize_response(resp.rootNode, analysis_results['root_node'])
        resp.tensor_utilization = float(analysis_results['tensor_core_perc'])
    except AnalysisError as ex:
        message = str(ex)
        logger.error(message)
        resp.analysis_error.error_message = message
    except Exception as ex:
        message = str(ex)
        logger.error(message)
        logger.error('There was an error measuring utilization')
        resp.analysis_error.error_message = (
            'There was an error measuring utilization')
    finally:
        return resp
