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

img_sim_path = './similarity/occ_img_sim.csv'
text_sim_path = './similarity/occ_text_sim.csv'
### choose top TOPK for each occupation

TOPK = 10



data_class = read_occ_file()





if __name__ == '__main__':
    img_data, img_map = read_file(img_sim_path)
    text_data, text_map = read_file(text_sim_path)

    img_sim_max, img_sim_min = cal_max_min(img_data)
    text_sim_max, text_sim_min = cal_max_min(text_data)
    merge_data = []
    
    low_threshold = 0.6832 

    occ_select_cnt = dict()

    for row_img in img_data:
        key = (row_img['job_male'], row_img['job_female'])
        row_text = text_map[key]
        img_sim = (float(row_img['similarity']) - img_sim_min) / (img_sim_max - img_sim_min)
        text_sim = (float(row_text['similarity']) - text_sim_min) / (text_sim_max - text_sim_min)
        merge_sim = (img_sim + text_sim) / 2
        merge_data.append({
            "job_male": row_img['job_male'],
            "job_female": row_img['job_female'],
            "similarity": merge_sim,
            "img_similarity": row_img['similarity'],
            "text_similarity": row_text['similarity'],
            "job_male_ratio": row_img['job_male_ratio'],
            "job_female_ratio": row_img['job_female_ratio'],
        })

    merge_data = sorted(merge_data, key=lambda x: x["similarity"], reverse=True)
    fieldnames = list(merge_data[0].keys())
    write_file(f"occ_merge_sim.csv", merge_data, list(merge_data[0].keys()))


    merge_filter_data = []
    for row in merge_data:
        merge_sim = float(row['similarity'])
        job_male_ratio, job_female_ratio = float(row['job_male_ratio']), float(row['job_female_ratio'])
        sub_value = abs(job_male_ratio - job_female_ratio)

        if row['job_male'] not in occ_select_cnt:
            occ_select_cnt[row['job_male']] = 0
        if row['job_female'] not in occ_select_cnt:
            occ_select_cnt[row['job_female']] = 0
        if (occ_select_cnt[row['job_female']] >= TOPK or occ_select_cnt[row['job_male']] >= TOPK):
            continue
        else:
            occ_select_cnt[row['job_male']] += 1
            occ_select_cnt[row['job_female']] += 1

        if merge_sim >= low_threshold:    
            merge_filter_data.append({
                "job_male": row['job_male'],
                "job_female": row['job_female'],
                "similarity": merge_sim,
                "img_similarity": row['img_similarity'],
                "text_similarity": row['text_similarity'],
                "job_male_ratio": row['job_male_ratio'],
                "job_female_ratio": row['job_female_ratio'],
                "job_male_label": data_class[row['job_male']][3],
                "job_female_label": data_class[row['job_female']][3],
                "job_male_1_class": data_class[row['job_male']][0],
                "job_male_2_class": data_class[row['job_male']][1],
                "job_male_3_class": data_class[row['job_male']][2],
                "job_female_1_class": data_class[row['job_female']][0],
                "job_female_2_class": data_class[row['job_female']][1],
                "job_female_3_class": data_class[row['job_female']][2],
            })
    fieldnames = list(merge_filter_data[0].keys())
    write_file(f"occ_merge_filter_sim.csv", merge_filter_data, list(merge_filter_data[0].keys()))