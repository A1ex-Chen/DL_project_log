def energy_compute(self) ->pm.EnergyResponse:
    energy_measurer = EnergyMeasurer()
    model = self._model_provider()
    inputs = self._input_provider()
    iteration = self._iteration_provider(model)
    resp = pm.EnergyResponse()
    try:
        energy_measurer.begin_measurement()
        iterations = 20
        for _ in range(iterations):
            iteration(*inputs)
        energy_measurer.end_measurement()
        resp.total_consumption = energy_measurer.total_energy() / float(
            iterations)
        resp.batch_size = self._batch_size
        components = []
        components_joules = []
        if energy_measurer.cpu_energy() is not None:
            cpu_component = pm.EnergyConsumptionComponent()
            cpu_component.component_type = pm.ENERGY_CPU_DRAM
            cpu_component.consumption_joules = energy_measurer.cpu_energy(
                ) / float(iterations)
            components.append(cpu_component)
            components_joules.append(cpu_component.consumption_joules)
        else:
            cpu_component = pm.EnergyConsumptionComponent()
            cpu_component.component_type = pm.ENERGY_CPU_DRAM
            cpu_component.consumption_joules = 0.0
            components.append(cpu_component)
            components_joules.append(cpu_component.consumption_joules)
        gpu_component = pm.EnergyConsumptionComponent()
        gpu_component.component_type = pm.ENERGY_NVIDIA
        gpu_component.consumption_joules = energy_measurer.gpu_energy(
            ) / float(iterations)
        components.append(gpu_component)
        components_joules.append(gpu_component.consumption_joules)
        resp.components.extend(components)
        path_to_entry_point = os.path.join(self._project_root, self.
            _entry_point)
        past_runs = (self._energy_table_interface.
            get_latest_n_entries_of_entry_point(10, path_to_entry_point))
        resp.past_measurements.extend(_convert_to_energy_responses(past_runs))
        current_entry = [path_to_entry_point] + components_joules
        current_entry.append(self._batch_size)
        self._energy_table_interface.add_entry(current_entry)
    except AnalysisError as ex:
        message = str(ex)
        logger.error(message)
        resp.analysis_error.error_message = message
    except Exception as ex:
        message = str(ex)
        logger.error(message)
        resp.analysis_error.error_message = (
            f'There was an error obtaining energy measurements: {message}')
    finally:
        return resp
