def _convert_to_energy_responses(entries: list) ->List[pm.EnergyResponse]:
    energy_response_list = []
    for entry in entries:
        if EnergyTableInterface.is_valid_entry_with_timestamp(entry):
            energy_response = pm.EnergyResponse()
            cpu_component = pm.EnergyConsumptionComponent()
            cpu_component.component_type = pm.ENERGY_CPU_DRAM
            cpu_component.consumption_joules = entry[1]
            gpu_component = pm.EnergyConsumptionComponent()
            gpu_component.component_type = pm.ENERGY_NVIDIA
            gpu_component.consumption_joules = entry[2]
            energy_response.total_consumption = (gpu_component.
                consumption_joules + cpu_component.consumption_joules)
            energy_response.components.extend([cpu_component, gpu_component])
            energy_response.batch_size = entry[3]
            energy_response_list.append(energy_response)
    return energy_response_list
