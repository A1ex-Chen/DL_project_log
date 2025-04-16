def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=
    False, echo=True) ->_RunOutput:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_stream_subprocess(cmd, env=env, stdin
        =stdin, timeout=timeout, quiet=quiet, echo=echo))
    cmd_str = ' '.join(cmd)
    if result.returncode > 0:
        stderr = '\n'.join(result.stderr)
        raise RuntimeError(
            f"""'{cmd_str}' failed with returncode {result.returncode}

The combined stderr from workers follows:
{stderr}"""
            )
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")
    return result
