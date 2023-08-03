아래는 주어진 내용을 Markdown 형식으로 이쁘게 변환한 것입니다:

## Pretrained Checkpoints

1. swin90.19_first_Train_iter_158000.pth  
   [Google Drive Link](https://drive.google.com/file/d/1TBosqsmo7mJKPXAA49pylf2W6kFhHNyr/view?usp=share_link)

2. swin90.19_second_Train_iter_158000.pth  
   [Google Drive Link](https://drive.google.com/file/d/1fI4Zfn_rKznJbPGee37jaEuZb_bO8fDK/view?usp=share_link)

3. Swin Pretrained pth  
   [Download Link](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth)

## How to start training

1. First Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/swin/swin90.19_first_Train_config.py
   ```

2. Second Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/swin/swin90.19_second_Train_config.py
   ```

## How to start inference

```
python work_dirs/swin/infer.py
```

위 내용을 .md 파일로 저장하면 Markdown 형식의 문서가 완성됩니다. 이렇게 수정된 내용을 참고하여 원하는 작업을 진행하시기 바랍니다.
