def save_opt_to_json(opt, conf_file):
    with open(conf_file, 'w', encoding='utf-8') as f:
        json.dump(opt, f, indent=4)
