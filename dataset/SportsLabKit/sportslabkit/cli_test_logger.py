def test_logger(self, log_level: str='INFO'):
    """Test the output of soccertrack's logger

        Args:
            log_level (str): The log level to use. Defaults to "INFO".
        """
    set_log_level(log_level)
    logger.debug("That's it, beautiful and simple logging!")
    logger.info('This is an info message')
    logger.success('success!')
    logger.warning('I am warning you Github copilot!')
    logger.error('I am error you Github copilot!')
    logger.critical('Fire in the hole!')
