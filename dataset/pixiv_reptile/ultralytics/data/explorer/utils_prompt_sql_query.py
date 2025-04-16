def prompt_sql_query(query):
    """Plots images with optional labels from a similar data set."""
    check_requirements('openai>=1.6.1')
    from openai import OpenAI
    if not SETTINGS['openai_api_key']:
        logger.warning(
            'OpenAI API key not found in settings. Please enter your API key below.'
            )
        openai_api_key = getpass.getpass('OpenAI API key: ')
        SETTINGS.update({'openai_api_key': openai_api_key})
    openai = OpenAI(api_key=SETTINGS['openai_api_key'])
    messages = [{'role': 'system', 'content':
        """
                You are a helpful data scientist proficient in SQL. You need to output exactly one SQL query based on
                the following schema and a user request. You only need to output the format with fixed selection
                statement that selects everything from "'table'", like `SELECT * from 'table'`

                Schema:
                im_file: string not null
                labels: list<item: string> not null
                child 0, item: string
                cls: list<item: int64> not null
                child 0, item: int64
                bboxes: list<item: list<item: double>> not null
                child 0, item: list<item: double>
                    child 0, item: double
                masks: list<item: list<item: list<item: int64>>> not null
                child 0, item: list<item: list<item: int64>>
                    child 0, item: list<item: int64>
                        child 0, item: int64
                keypoints: list<item: list<item: list<item: double>>> not null
                child 0, item: list<item: list<item: double>>
                    child 0, item: list<item: double>
                        child 0, item: double
                vector: fixed_size_list<item: float>[256] not null
                child 0, item: float

                Some details about the schema:
                - the "labels" column contains the string values like 'person' and 'dog' for the respective objects
                    in each image
                - the "cls" column contains the integer values on these classes that map them the labels

                Example of a correct query:
                request - Get all data points that contain 2 or more people and at least one dog
                correct query-
                SELECT * FROM 'table' WHERE  ARRAY_LENGTH(cls) >= 2  AND ARRAY_LENGTH(FILTER(labels, x -> x = 'person')) >= 2  AND ARRAY_LENGTH(FILTER(labels, x -> x = 'dog')) >= 1;
             """
        }, {'role': 'user', 'content': f'{query}'}]
    response = openai.chat.completions.create(model='gpt-3.5-turbo',
        messages=messages)
    return response.choices[0].message.content
