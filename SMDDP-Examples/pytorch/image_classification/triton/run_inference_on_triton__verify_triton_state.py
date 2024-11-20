def _verify_triton_state(self, triton_client):
    errors = []
    if not triton_client.is_server_live():
        errors.append(f'Triton server {self._server_url} is not live')
    elif not triton_client.is_server_ready():
        errors.append(f'Triton server {self._server_url} is not ready')
    elif not triton_client.is_model_ready(self._model_name, self._model_version
        ):
        errors.append(
            f'Model {self._model_name}:{self._model_version} is not ready')
    return errors
