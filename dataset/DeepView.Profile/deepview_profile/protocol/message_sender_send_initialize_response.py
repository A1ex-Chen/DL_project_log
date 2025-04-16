def send_initialize_response(self, context):
    message = pm.InitializeResponse()
    connection = self._connection_manager.get_connection(context.address)
    message.server_project_root = connection.project_root
    message.entry_point.components.extend(connection.entry_point.split(os.sep))
    message.hardware.hostname = platform.node()
    message.hardware.os = ' '.join(list(platform.uname()))
    pynvml.nvmlInit()
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        message.hardware.gpus.append(device_name)
    pynvml.nvmlShutdown()
    self._send_message(message, 'initialize', context)
