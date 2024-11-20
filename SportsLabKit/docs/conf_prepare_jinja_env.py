def prepare_jinja_env(jinja_env) ->None:
    """Add `contains` custom test to Jinja environment."""
    jinja_env.tests['contains'] = contains
