def setup(app):
    app.connect('autoapi-skip-member', autoapi_skip_members)
