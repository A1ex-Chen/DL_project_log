import csv
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import torch
import clip
from PIL import Image
from tqdm import tqdm

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

test_root = "xl_generate_base" # the path of the base image data









if __name__=='__main__':
    
    feature_path = "features.pkl"
    all_occ_list, job_tend_to_male, job_tend_to_female = read_occ_file()
    
    job_male_text = list(job_tend_to_male.keys())
    print(job_male_text)
    print(job_tend_to_male)
    job_female_text = list(job_tend_to_female.keys())

    if os.path.exists(feature_path):
        with open(feature_path, "rb") as f:
            features = pickle.load(f)
            m_f_fea, m_m_fea = features["m_f_fea"], features["m_m_fea"]
            f_f_fea, f_m_fea = features["f_f_fea"], features["f_m_fea"]
            m_f_fea = m_f_fea.to(device)
            m_m_fea = m_m_fea.to(device)
            f_f_fea = f_f_fea.to(device)
            f_m_fea = f_m_fea.to(device)
    else:
        job_tend_to_male_features, m_f_fea, m_m_fea = get_occ_features(job_male_text)
        job_tend_to_female_features, f_f_fea, f_m_fea = get_occ_features(job_female_text)

        with open(feature_path, "wb") as f:
            pickle.dump({
                "m_f_fea": m_f_fea.cpu(),
                "m_m_fea": m_m_fea.cpu(),
                "f_f_fea": f_f_fea.cpu(),
                "f_m_fea": f_m_fea.cpu()
            }, f)

    # cal the sim of occ 
    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits_sim_f = m_f_fea @ f_f_fea.t()
        logits_sim_m = m_m_fea @ f_m_fea.t()
        
    logits_avg = torch.zeros_like(logits_sim_f)  

    for i in range(logits_avg.shape[0]):
        for j in range(logits_avg.shape[1]):
            if logits_sim_f[i, j] == 0:
                logits_avg[i, j] = logits_sim_m[i, j]
            elif logits_sim_m[i, j] == 0:
                logits_avg[i, j] = logits_sim_f[i, j]
            else:
                logits_avg[i, j] = (logits_sim_f[i, j] + logits_sim_m[i, j]) / 2

    result_map = {}
    for i in range(0, len(job_male_text)):
        for j in range(0, len(job_female_text)):
            result_map[(job_male_text[i], job_female_text[j])] = logits_avg[i,j].item()
    sorted_result = sorted(result_map.items(), key=lambda x: x[1], reverse=True)

    data_to_out = []
    for item in sorted_result:
        data_to_out.append({
            "job_male": item[0][0],
            "job_female": item[0][1],
            "similarity": item[1],
            "job_male_ratio": job_tend_to_male[item[0][0]],
            "job_female_ratio": job_tend_to_female[item[0][1]]
        })
    fieldnames = list(data_to_out[0].keys())
    os.makedirs("./similarity", exist_ok=True)
    write_file(f"./similarity/occ_img_sim.csv", data_to_out, list(data_to_out[0].keys()))

    