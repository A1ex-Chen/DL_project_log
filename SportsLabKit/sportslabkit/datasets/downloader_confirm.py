def confirm(msg: str) ->bool:
    """Confirm the user input."""
    logger.info(msg + ' [y/n]')
    val = input()
    logger.info(f'You entered: {val}')
    if val.lower() in ['y', 'yes']:
        return True
    elif val.lower() in ['n', 'no']:
        return False
    else:
        logger.error('Invalid input. Please try again.')
        return confirm(msg)
