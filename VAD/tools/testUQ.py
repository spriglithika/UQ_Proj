# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys

sys.path.append("")
import argparse
import mmcv
import os
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed

# from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

import warnings
import threading
import copy
LOSS_PATH = '../outs/losses'

warnings.filterwarnings("ignore")
##################################################################################################################################
def save_tensor(tensor_path_pairs:dict):
    """
    Args:
        tensor_path_pairs (dict): {path:tensor}
    """
    for path, emb in tensor_path_pairs.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(emb.cpu(), path)
        print(f"Saved {path}")
##################################################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--json_dir", help="json parent dir name file"
    )  # NOTE: json file parent folder name
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    # samples_per_gpu = 1
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    samples_per_gpu = 1
    if isinstance(cfg.data.train, dict):
        cfg.data.train.test_mode = True
        # samples_per_gpu = cfg.data.train.pop("samples_per_gpu", 1)
        # if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    # init distributed env first, since logger depends on the dist info.
    samples_per_gpu = 1
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=samples_per_gpu,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False,
    #     nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    # )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # if not distributed:
        # assert False
    # model = MMDataParallel(model, device_ids=[0])
    # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    def get_UQ_data(dataset, data_handle):
        model.cuda()
        model.eval()
        data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        def step(
            points=None,
            img_metas=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            map_gt_bboxes_3d=None,
            map_gt_labels_3d=None,
            gt_labels=None,
            gt_bboxes=None,
            img=None,
            proposals=None,
            gt_bboxes_ignore=None,
            map_gt_bboxes_ignore=None,
            img_depth=None,
            img_mask=None,
            ego_his_trajs=None,
            ego_fut_trajs=None,
            ego_fut_masks=None,
            ego_fut_cmd=None,
            ego_lcf_feat=None,
            gt_attr_labels=None,
                **kwargs):
            # print(len(img))
            # print(len(img[0].data))
            img = torch.tensor(img.data).cuda()
            print(img.shape)
            # print(img_metas)
            img_metas = img_metas[0].data[0]
            len_queue = img.size(0)
            prev_img = img[:,:-1, ...]
            img = img[:,-1, ...]

            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = model.obtain_history_bev(prev_img, prev_img_metas)
            # import pdb;pdb.set_trace()
            # prev_bev = (
                # model.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None
            # )
            # img_metas = [each[len_queue - 1] for each in img_metas]
            img_feats = model.extract_feat(img=img.squeeze(0), img_metas=img_metas)
            variants = (data_handle, e)
            outs = model.pts_bbox_head(
                img_feats,
                img_metas,
                prev_bev=None,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                variants = variants
            )
            batch, num_agent = outs["all_traj_preds"][-1].shape[:2]
            print(ego_his_trajs[0].data[0][None,...])
            planning_dict = model.pts_bbox_head.loss_planning(
                outs["ego_fut_preds"],
                ego_fut_trajs[0].data[0].squeeze(1),#[None,...],
                ego_fut_masks[0].data[0].squeeze(1).squeeze(1),#[None,...],
                ego_fut_cmd[0].data[0].squeeze(1).squeeze(1),#[None,...],
                outs["map_all_pts_preds"][-1],
                outs["map_all_cls_scores"][-1].sigmoid(),
                outs["all_bbox_preds"][-1][...,0:2],
                outs["all_traj_preds"][-1].view(batch, num_agent, 6, 6, 2),
                outs["all_cls_scores"][-1].sigmoid(),
                outs["all_traj_cls_scores"][-1].view(batch, num_agent, 6),
                variants=variants
            )
            # loss_path = LOSS_PATH + '/' + variants[0] + '/' + str(variants[1]) + '.pth'
            # thread = threading.Thread(target=save_tensor, args = {loss_path:planning_dict.detach().clone()})
            # thread.start()
        prog_bar = mmcv.ProgressBar(len(dataset))
        for e, data in enumerate(data_loader):
            # print(data.keys()) #
            # img_metas = data['img_metas']
            # img = data['img']
            # ego_his_trajs = data['ego_his_trajs']
            # ego_fut_masks = data['ego_fut_masks']
            # ego_fut_cmd = data['ego_fut_cmd']
            # print(img)
            # print(torch.tensor(img).shape)
            # print(data.keys())
            step(**data)
            batch_size = len(data)
            for _ in range(batch_size):
                prog_bar.update()
    print('VAL:')
    get_UQ_data(build_dataset(cfg.data.test),'val')
    print('TRAIN:')
    get_UQ_data(build_dataset(cfg.data.train),'train')

    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False,
    #     )
    #     outputs = custom_multi_gpu_test(
    #         model, data_loader, args.tmpdir, args.gpu_collect
    #     )

    # tmp = {}
    # tmp["bbox_results"] = outputs
    # outputs = tmp
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     if args.out:
    #         print(f"\nwriting results to {args.out}")
    #         # assert False
    #         if isinstance(outputs, list):
    #             mmcv.dump(outputs, args.out)
    #         else:
    #             mmcv.dump(outputs["bbox_results"], args.out)
    #     kwargs = {} if args.eval_options is None else args.eval_options
    #     kwargs["jsonfile_prefix"] = osp.join(
    #         "test",
    #         args.config.split("/")[-1].split(".")[-2],
    #         time.ctime().replace(" ", "_").replace(":", "_"),
    #     )
    #     if args.format_only:
    #         dataset.format_results(outputs["bbox_results"], **kwargs)

    #     if args.eval:
    #         eval_kwargs = cfg.get("evaluation", {}).copy()
    #         # hard-code way to remove EvalHook args
    #         for key in [
    #             "interval",
    #             "tmpdir",
    #             "start",
    #             "gpu_collect",
    #             "save_best",
    #             "rule",
    #         ]:
    #             eval_kwargs.pop(key, None)
    #         eval_kwargs.update(dict(metric=args.eval, **kwargs))

    #         print(dataset.evaluate(outputs["bbox_results"], **eval_kwargs))

        # # # NOTE: record to json
        # json_path = args.json_dir
        # if not os.path.exists(json_path):
        #     os.makedirs(json_path)

        # metric_all = []
        # for res in outputs['bbox_results']:
        #     for k in res['metric_results'].keys():
        #         if type(res['metric_results'][k]) is np.ndarray:
        #             res['metric_results'][k] = res['metric_results'][k].tolist()
        #     metric_all.append(res['metric_results'])

        # print('start saving to json done')
        # with open(json_path+'/metric_record.json', "w", encoding="utf-8") as f2:
        #     json.dump(metric_all, f2, indent=4)
        # print('save to json done')


if __name__ == "__main__":
    main()
