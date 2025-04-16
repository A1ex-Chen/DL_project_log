def deprecation_warn(arg, new_arg):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    LOGGER.warning(
        f"WARNING ⚠️ '{arg}' is deprecated and will be removed in in the future. Please use '{new_arg}' instead."
        )
