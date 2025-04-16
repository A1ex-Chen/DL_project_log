def show_authenticate_message() ->Any:
    """Show the instructions to authenticate the Kaggle API key."""
    logger.info('Please authenticate with your Kaggle account.')
    has_account = confirm('Do you have a Kaggle account?')
    if has_account:
        platform = get_platform()
        username = prompt('Please enter your kaggle username', type=str)
        logger.info(
            f'Please go to https://www.kaggle.com/{username}/account and follow these steps:'
            )
        logger.info('1. Scroll and click the "Create API Token" section.')
        logger.info('2. A file named "kaggle.json" will be downloaded.')
        if platform in ['linux', 'mac']:
            logger.info('3. Move the file to ~/.kaggle/kaggle.json')
        elif platform == 'windows':
            logger.info(
                '3. Move the file to C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json'
                )
        else:
            logger.info(
                '3. Move the file to ~/.kaggle/kaggle.json  folder in Mac and Linux or to C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json  on windows.'
                )
        if not confirm('Have you completed the steps above? Type N to abort.'):
            logger.info('Aborting.')
            return None
        return authenticate(show_message=False)
    logger.info(
        'Please create a Kaggle account and follow the instructions on the following:'
        )
    logger.info('https://www.kaggle.com/')
    return None
