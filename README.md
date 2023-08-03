# InternImage

[Install](https://github.com/OpenGVLab/InternImage/blob/master/segmentation/README.md)

## Pretrained Checkpoints

1. INTERN_best_mDice_iter_336000.pth  
   [Download Link](https://drive.google.com/file/d/1B2ieT2W_I-tp-yjzfR7_sOoKfQK8sj7R/view?usp=drive_link)

2. INTERN_k1_best_mDice_iter_210000.pth  
   [Download Link](https://drive.google.com/file/d/1X-1BNwz0eQ4AV2eqtzdBtenS2i5bPAyW/view?usp=drive_link)

3. INTERN_k3_best_mDice_iter_220000.pth  
   [Download Link](https://drive.google.com/file/d/1MvktQmj8wdS1PNDlLoPUvT4b4zOxyAEN/view?usp=sharing)

4. INTERN_k4_best_mDice_iter_280000.pth  
   [Download Link](https://drive.google.com/file/d/1ijjzE5yT190OEeIHdn1_t27Q-y0Mk7vO/view?usp=drive_link)

## How to start training

```
   cd mmseg_0.x.x
   python segmentation/train.py work_dirs/no4/INTERN_config.py
```


## How to start inference

```
   cd mmseg_0.x.x
   python segmentation/inference.py
```


# SWINv2

[Install](mmseg_1.x.x/docs/en/get_started.md)

## Pretrained path

1. Swin Pretrained pth  
   [Download Link](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth)

## Pretrained Checkpoints

1. swin_best_mDice_iter_320000.pth (160k + 160k)   
   [Google Drive Link](https://drive.google.com/file/d/1fI4Zfn_rKznJbPGee37jaEuZb_bO8fDK/view?usp=drive_link)

## How to start training

1. Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/swin/swin_config.py
   ```

## How to start inference

```
python work_dirs/swin/infer.py
```



# Mask2Former

[Install](mmseg_1.x.x/docs/en/get_started.md)

## Pretrained Checkpoints

1. m2f_K2_best_mDice_iter_90000.pth  (90k)  
   [Google Drive Link](https://drive.google.com/file/d/1NHI02wH_hzVNtEsWXzRa23MlH903W7an/view?usp=drive_link)

2. m2f_K3_best_mDice_iter_90000.pth (90k + 90k)    
   [Google Drive Link](https://drive.google.com/file/d/199AWAwd8n758zGUQIJlDVl_wd3dz1AwT/view?usp=drive_link)

3. m2f_K4_best_mDice_iter_90000.pth (90k + 90k)    
   [Google Drive Link](https://drive.google.com/file/d/1C637rkVaIV14SmdTHMzFM4SYQp-ceY6e/view?usp=drive_link)

## How to start training

1. K2 Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/mask2former/m2f_config_k2.py
   ```

2. K3 Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/mask2former/m2f_config_k3.py
   ```
      
3. K4 Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/mask2former/m2f_config_k4.py
   ```
   
## How to start inference

```
python work_dirs/mask2former/infer_m2f.py
```


# Ensemble

1. Swin (단일 모델)
2. internimage 
   - best_mDice_iter_336000 + k1 + k3 + k4 (threshold = 2)
3. mask2former
   - k2 + k3 + k4 (threshold = 2)

-> last submit  
swin + internimage + mask2former (threshold = 2)
