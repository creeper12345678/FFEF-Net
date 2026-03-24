_base_ = [
    '../_base_/models/upernet_convnext-S1.py', '../_base_/datasets/settlement.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_town.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'   # noqa
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ConvNeXtSpectral',
        arch='base',
        in_channels=7,
        out_indices=[0, 1, 2, 3], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
   
   decode_head=dict(
        type='FreqUPerHead',  # 自定义 解码
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3], 
        channels=512,
        dropout_ratio=0.4,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        freq_fusion_cfg=dict(
            lowpass_kernel=5,  
            highpass_kernel=5, 
            compressed_channels=256,  
            feature_resample=True, 
            comp_feat_upsample=True,
            use_high_pass=True,
            use_low_pass=True,
            hr_residual=True,
            hamming_window=True,
            align_corners=True,
           )),

   
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,  
        in_index=2,
        channels=512,
        concat_input=False,
        dropout_ratio=0.4,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
