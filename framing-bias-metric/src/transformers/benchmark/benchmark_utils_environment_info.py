@property
def environment_info(self):
    if self._environment_info is None:
        info = {}
        info['transformers_version'] = version
        info['framework'] = self.framework
        if self.framework == 'PyTorch':
            info['use_torchscript'] = self.args.torchscript
        if self.framework == 'TensorFlow':
            info['eager_mode'] = self.args.eager_mode
            info['use_xla'] = self.args.use_xla
        info['framework_version'] = self.framework_version
        info['python_version'] = platform.python_version()
        info['system'] = platform.system()
        info['cpu'] = platform.processor()
        info['architecture'] = platform.architecture()[0]
        info['date'] = datetime.date(datetime.now())
        info['time'] = datetime.time(datetime.now())
        info['fp16'] = self.args.fp16
        info['use_multiprocessing'] = self.args.do_multi_processing
        info['only_pretrain_model'] = self.args.only_pretrain_model
        if is_psutil_available():
            info['cpu_ram_mb'] = bytes_to_mega_bytes(psutil.virtual_memory(
                ).total)
        else:
            logger.warning(
                "Psutil not installed, we won't log available CPU memory.Install psutil (pip install psutil) to log available CPU memory."
                )
            info['cpu_ram_mb'] = 'N/A'
        info['use_gpu'] = self.args.is_gpu
        if self.args.is_gpu:
            info['num_gpus'] = 1
            if is_py3nvml_available():
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                info['gpu'] = nvml.nvmlDeviceGetName(handle)
                info['gpu_ram_mb'] = bytes_to_mega_bytes(nvml.
                    nvmlDeviceGetMemoryInfo(handle).total)
                info['gpu_power_watts'
                    ] = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                info['gpu_performance_state'
                    ] = nvml.nvmlDeviceGetPerformanceState(handle)
                nvml.nvmlShutdown()
            else:
                logger.warning(
                    "py3nvml not installed, we won't log GPU memory usage. Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                info['gpu'] = 'N/A'
                info['gpu_ram_mb'] = 'N/A'
                info['gpu_power_watts'] = 'N/A'
                info['gpu_performance_state'] = 'N/A'
        info['use_tpu'] = self.args.is_tpu
        self._environment_info = info
    return self._environment_info
