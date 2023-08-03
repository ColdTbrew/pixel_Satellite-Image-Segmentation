import os
import mmcv
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from mmseg.apis import init_segmentor, inference_segmentor

import pandas as pd
import numpy as np
import json
import numpy as np

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main(args):
    save_dir = args.save_dir
    file_name = args.file_name
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    sample_path = args.sample_path
    test_image_path = args.test_image_path
    device = args.device
    print('save_dir : ', save_dir)
    print('file_name : ', file_name)
    print('config_file : ', config_file)
    print('checkpoint_file : ', checkpoint_file)
    print('sample_path : ', sample_path)
    print('test_image_path : ', test_image_path)
                
    model = init_segmentor(config_file, checkpoint_file, device)
    data = pd.read_csv(sample_path)['img_id'].values.tolist()
    
    with torch.no_grad():
        model.eval()
        result = []
        for img_id in tqdm(data):
            img_path = os.path.join(test_image_path, img_id + ".png")
            mask = inference_segmentor(model, img_path)
            # mask = mask.pred_sem_seg.data
            mask = np.array(mask)
            mask_rle = rle_encode(mask)
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)

    submit = pd.read_csv(sample_path)
    submit['mask_rle'] = result
    submit.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default = "/home/jovyan/work/work_space/JSJ/submit/mmseg")
    parser.add_argument('--file_name', type=str, default = 'InternImage_Plus4_332000')
    parser.add_argument('--config_file', type=str, default = '/home/jovyan/work/work_space/JSJ/InternImage/segmentation/work_dirs/InternImage/PLUS/Fourth/InternImage_plus.py')
    parser.add_argument('--checkpoint_file', type=str, default = '/home/jovyan/work/work_space/JSJ/InternImage/segmentation/work_dirs/InternImage/best_mDice_iter_92000.pth')
    parser.add_argument('--sample_path', type=str, default = "/home/jovyan/work/work_space/JSJ/submit/sample_submission.csv")
    parser.add_argument('--test_image_path', type=str, default = "/home/jovyan/work/datasets/satellite/images/test")
    parser.add_argument('--device', type=str, default = "cuda")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
