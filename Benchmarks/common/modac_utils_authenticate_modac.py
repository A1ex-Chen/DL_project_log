def authenticate_modac(generate_token=False):
    """
    Authenticates a user on modac.cancer.gov

        Parameters
        ----------
        generate_token : Bool
            Either generate a new token, or read saved token if it exists


        Returns
        ----------
        tuple(string,string)
            tuple with the modac credentials
    """
    from os.path import expanduser
    home = expanduser('~')
    modac_token_dir = os.path.abspath(os.path.join(home, '.nci-modac'))
    modac_token_file = 'credentials.json'
    user_attr = 'modac_user'
    token_attr = 'modac_token'
    modac_token_path = os.path.join(modac_token_dir, modac_token_file)
    credentials_dic = {}
    if not generate_token and os.path.exists(modac_token_path):
        with open(modac_token_path) as f:
            credentials_dic = json.load(f)
    else:
        modac_user = input('MoDaC Username: ')
        import getpass
        modac_pass = getpass.getpass('MoDaC Password: ')
        auth = modac_user, modac_pass
        auth_url = 'https://modac.cancer.gov/api/authenticate'
        print('Authenticating ' + modac_user + ' ...')
        response = requests.get(auth_url, auth=auth, stream=True)
        if response.status_code != 200:
            print('Error authenticating modac user:{0}', modac_user)
            raise Exception('Response code: {0}, Response message: {1}'.
                format(response.status_code, response.text))
        else:
            token = response.text
            if not os.path.exists(modac_token_path):
                save_question = 'Save MoDaC token in {0}'.format(
                    modac_token_path)
                save_token = query_yes_no(save_question)
            else:
                save_token = True
            if save_token:
                if not os.path.isdir(modac_token_dir):
                    os.mkdir(modac_token_dir)
                credentials_dic[user_attr] = modac_user
                credentials_dic[token_attr] = token
                with open(modac_token_path, 'w') as outfile:
                    json.dump(credentials_dic, outfile, indent=4)
    return credentials_dic[user_attr], credentials_dic[token_attr]
