import csv
import os
import clip







if __name__=="__main__":
    path = "../../resources/occ_us.csv"
    data, job_tend_to_male, job_tend_to_female = read_file(path)

    clip_model, preprocess = setup_clip_model("cuda")

    job_male_text = list(job_tend_to_male.keys())
    job_female_text = list(job_tend_to_female.keys())

    job_male_feature = get_text_feature(job_male_text, clip_model)
    job_female_feature = get_text_feature(job_female_text, clip_model)

    logits = job_male_feature @ job_female_feature.t()
    print(logits)

    result_map = {}
    for i in range(0, len(job_male_text)):
        for j in range(0, len(job_female_text)):
            result_map[(job_male_text[i], job_female_text[j])] = logits[i,j].item()
    sorted_result = sorted(result_map.items(), key=lambda x: x[1], reverse=True)
    


    data_to_out = []
    for item in sorted_result:
        sub_value = abs(job_tend_to_female[item[0][1]]-job_tend_to_male[item[0][0]])
        
        data_to_out.append({
            "job_male": item[0][0],
            "job_female": item[0][1],
            "similarity": item[1],
            "job_male_ratio": job_tend_to_male[item[0][0]],
            "job_female_ratio": job_tend_to_female[item[0][1]]
        })
    fieldnames = list(data_to_out[0].keys())
    os.makedirs("./similarity", exist_ok=True)
    write_file(f"./similarity/socc_text_sim.csv", data_to_out, list(data_to_out[0].keys()))

