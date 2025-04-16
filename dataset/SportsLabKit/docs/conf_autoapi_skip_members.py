def autoapi_skip_members(app, what, name, obj, skip, options):
    if what == 'module':
        skip = True
    elif what == 'data':
        if obj.name in ['EASING_FUNCTIONS', 'ParamType']:
            skip = True
    elif what == 'function':
        if obj.name in ['working_directory']:
            skip = True
    return skip
