def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = 'https://api.openai.com/v1/moderations'
    headers = {'Content-Type': 'application/json', 'Authorization': 
        'Bearer ' + os.environ['OPENAI_API_KEY']}
    text = text.replace('\n', '')
    data = '{' + '"input": ' + f'"{text}"' + '}'
    data = data.encode('utf-8')
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()['results'][0]['flagged']
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False
    return flagged
