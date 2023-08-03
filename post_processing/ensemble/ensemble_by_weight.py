import os, cv2
import mmcv
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from mmseg.apis import init_model, inference_model

import pandas as pd
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def func(data):
    min_val = min(data)
    data = [np.log(x - min_val + 1) + 1 for x in data]
    a = np.sum(data)
    data = [x / a for x in data]
    return data

def normalize_scores(scores):
    max_score = max(scores)
    normalized_scores = [score / max_score for score in scores]
    return normalized_scores

source_dir = "source"
sample_submission_dir = "sample_submission.csv"
save_dir = "submit_csv"
file_name = "ensemble6_11_th0.70.csv"  # 변수
threshold = 0.7  # 변수
print('threshold: ', threshold)
print('file_name : ',file_name)

img_shape = (224, 224)

# 파일 이름을 기준으로 정렬하여 파일 리스트 가져오기
file_list = sorted([f for f in os.listdir(source_dir) if os.path.splitext(f)[1] == '.csv'])

source_list = [pd.read_csv(os.path.join(source_dir, f)) for f in file_list]

result = []
#target csv:  ['InternImage_Plus4_332000.csv', 'ensemble4_deep+intern+hr+swin_Th2.csv', 'ensemble4_intern+hr+swin_9019.csv', 'ensemble5_intern+mask2former+swin_9019.csv', 'submit_swin2_90.19_01.csv']

# weights = [ 81,29, 81.59, 81.83, 82.04,   81.32]

#Record
# # target csv : ['InternImage3.csv', 'InternImage_Plus3_2.csv', 'InternImage_Plus4_332000.csv', 
# # 'ensemble4_deep+intern+hr+swin-Copy1.csv', 'ensemble4_deep+intern+hr+swin_Th2.csv', 'ensemble4_intern+hr+swin_9019.csv', 'ensemble5_intern+mask2former+swin_9019.csv', 'submit_hrnet_03.csv', 'submit_m2f_k2.csv', 'submit_swin2_90.19_01.csv', 'submit_swin_89.74_03.csv']

weights = [79.99, 80.74, 81,29,80.64, 81.59, 81.83, 82.04, 79.69, 80.14, 81.32, 80.78]

print(func(weights))
table = pd.read_csv(sample_submission_dir)["img_id"].values.tolist()
print("target csv: ", file_list)

for img_id in tqdm(table):
    total_mask = np.zeros((224, 224))
    idx = 0
    for f in source_list:
        _df = f[f["img_id"] == img_id]["mask_rle"].values[0]
        mask = rle_decode(_df, (224, 224))
        mask = mask * func(weights)[idx]
        idx += 1
        total_mask += mask
    total_mask[total_mask < threshold] = 0
    total_mask[total_mask >= threshold] = 1

    mask_rle = rle_encode(total_mask)
    if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
        result.append(-1)
    else:
        result.append(mask_rle)

submit = pd.read_csv(sample_submission_dir)
submit['mask_rle'] = result
submit.to_csv(os.path.join(save_dir, file_name), index=False)
