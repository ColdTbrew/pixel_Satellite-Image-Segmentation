_base_ = [
    'upernet_vit-b16_ln_mln_m.py',
    'schedule_160k_m.py', 
    'default_runtime_m.py',
    'satellite.py'
]
#원래 이름 vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py
crop_size = (768, 768) #256->512
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
#pretrained='/home/jovyan/work/work_space/shc/mmsegmentation/configs/_myconfig_/pretrain/vit_base_patch16_224.pth','
    pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k/upernet_vit-b16_ln_mln_512x512_160k_ade20k_20210621_172828-f444c077.pth',
    backbone=dict(drop_path_rate=0.1, final_norm=True),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)#4->8
val_dataloader = dict(batch_size=1)#2->4
test_dataloader = val_dataloader
