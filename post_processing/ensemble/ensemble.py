import os, cv2
from tqdm import tqdm

import pandas as pd
import numpy as np

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

def min_max_scaler(data):
    min_val = min(data)
    max_val = max(data)
    scaled_data = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled_data

def main():
    source_dir = "source" #앙상블할 csv 한폴더에 넣기
    sample_submission_dir = "sample_submission.csv" #sample 위치
    test_img_dir = "images/test"
    save_dir = "submits_csv"
    file_name = "ensemble_last.csv" #변수
    threshold = 2   #변수
    

    source_list = [pd.read_csv(os.path.join(source_dir, f)) for f in os.listdir(source_dir) if os.path.splitext(f)[1] == '.csv']
    
    result = []
    weigths = []
    table = pd.read_csv(sample_submission_dir)["img_id"].values.tolist()
    print("target csv: ", [f for f in os.listdir(source_dir) if os.path.splitext(f)[1] == '.csv'])
    for img_id in tqdm(table):
        img_path = os.path.join(test_img_dir, img_id + ".png")
        img = cv2.imread(img_path)
        total_mask = np.zeros((img.shape[0], img.shape[1]))
        for f in source_list:
            _df = f[f["img_id"]==img_id]["mask_rle"].values[0]
            mask = rle_decode(_df, (img.shape[0], img.shape[1]))
            total_mask += mask
        total_mask[total_mask < threshold] = 0
        total_mask[total_mask >= threshold] = 1
            
        mask_rle = rle_encode(total_mask)
        if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
            result.append(-1)
        else:
            result.append(mask_rle)
            
    submit = pd.read_csv(sample_submission_dir)
    submit['mask_rle'] = result
    submit.to_csv(os.path.join(save_dir, file_name), index=False)
    
if __name__ == "__main__":
    main()
