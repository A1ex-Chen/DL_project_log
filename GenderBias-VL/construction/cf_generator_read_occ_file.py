def read_occ_file(self):
    data = []
    prompt_data = []
    occ_gender_prompts_map = {}
    with open(prompt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if len(line.strip()) == 0:
                continue
            str_ = str(line.strip('"').strip())
            prompt_data.append(str_)
    sublists = [prompt_data[i:i + 5] for i in range(0, len(prompt_data), 5)]
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        index = 0
        for row in reader:
            data.append(row['occupation'])
            occ_gender_prompts_map[row['occupation'], 'female'] = sublists[
                index]
            index += 1
            occ_gender_prompts_map[row['occupation'], 'male'] = sublists[index]
            index += 1
            if index == len(sublists):
                break
    return data, occ_gender_prompts_map
