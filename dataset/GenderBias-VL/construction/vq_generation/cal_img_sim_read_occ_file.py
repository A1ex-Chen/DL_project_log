def read_occ_file():
    path = '../../resources/occ_us.csv'
    data = []
    job_tend_to_male = {}
    job_tend_to_female = {}
    job_no_tend = {}
    cnt = 0
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row['occupation'])
            job = row['occupation']
            woman_ratio = row['Women']
            if woman_ratio == '-':
                print(f'skip {job} {woman_ratio}')
                job_no_tend[job] = 50
                continue
            woman_ratio = float(woman_ratio)
            if woman_ratio == 50:
                print(f'go to female {job} {woman_ratio}')
            if woman_ratio < 50:
                job_tend_to_male[job] = woman_ratio
            elif woman_ratio >= 50:
                job_tend_to_female[job] = woman_ratio
    return data, job_tend_to_male, job_tend_to_female
