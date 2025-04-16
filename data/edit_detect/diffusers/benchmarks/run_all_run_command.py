def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, 'decode'):
                output = output.decode('utf-8')
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"""Command `{' '.join(command)}` failed with the following error:

{e.output.decode()}"""
            ) from e
