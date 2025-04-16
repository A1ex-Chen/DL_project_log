def prompt(msg: str, type: Any) ->Any:
    """Prompt the user for input."""
    logger.info(msg)
    val = input()
    logger.info(f'You entered: {val}')
    try:
        return type(val)
    except ValueError:
        logger.error('Invalid input. Please try again.')
        return prompt(msg, type)
