def save(filename, msg):
    """Save log message"""
    path = os.getenv('TURBINE_OUTPUT')
    with open(path + '/' + filename, 'w') as file_json:
        file_json.write(json.dumps(msg, indent=4, separators=(',', ': ')))
