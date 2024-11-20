def read_file(path):
    data = []
    job_tend_to_male = {}
    job_tend_to_female = {}
    job_no_tend = {}
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
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
    print(f'job_no_tend: {len(job_no_tend)}')
    print(f'job_tend_to_male: {len(job_tend_to_male)}')
    print(f'job_tend_to_female: {len(job_tend_to_female)}')
    return data, job_tend_to_male, job_tend_to_female
