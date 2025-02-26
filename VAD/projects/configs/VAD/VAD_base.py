_base_ = ["VAD_base_e2e.py"]

# we have to change this according to their docs
# https://github.com/wljungbergh/VAD/blob/main/docs/train_eval.md
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

inference_pipeline = [
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="RandomScaleImageMultiViewImage", scales=[0.8]),
    dict(type="PadMultiViewImage", size_divisor=32),
]
