import copy
import uuid
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from nuscenes.eval.common.utils import (
    quaternion_yaw,
)
from pyquaternion import Quaternion

from projects.mmdet3d_plugin.VAD.modules.collision_optimization import (
    CollisionNonlinearOptimizer,
)
from projects.mmdet3d_plugin.VAD.VAD import VAD
from projects.mmdet3d_plugin.VAD.VAD_head import save_tensor, MOTION_EMB_PATH, LOSS_PATH
import threading

NUSCENES_CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

COLLISION_CLASSES_TO_CHECK = [
    "car",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "truck",
    "trailer",
]


@dataclass
class VADInferenceInput:
    imgs: np.ndarray
    """shape: (n-cams (6), h (900), w (1600) c (3)) | images without any preprocessing. should be in RGB order as uint8"""
    lidar_pose: np.ndarray
    """shape: (3, 4) | lidar pose in global frame"""
    lidar2img: np.ndarray
    """shape: (n-cams (6), 4, 4) | lidar2img transformation matrix, i.e., lidar2cam @ camera2img"""
    timestamp: float
    """timestamp of the current frame in seconds"""
    can_bus_signals: np.ndarray
    """shape: (16,) | see above for details"""
    command: int
    """0: right, 1: left, 2: straight"""


@dataclass
class VADAuxOutputs:
    objects_in_bev: Optional[List[List[float]]] = None  # N x [x, y, width, height, yaw]
    object_classes: Optional[List[str]] = None  # (N, )
    object_scores: Optional[List[float]] = None  # (N, )
    segmentation: Optional[List[List[float]]] = None
    seg_grid_centers: Optional[
        List[List[List[float]]]
    ] = None  # bev_h (200), bev_w (200), 2 (x & y)
    future_trajs: Optional[List[List[List[List[float]]]]] = None  # (N, 6, 6, 2)

    def to_json(self) -> dict:
        return dict(
            objects_in_bev=self.objects_in_bev,
            object_classes=self.object_classes,
            object_scores=self.object_scores,
            segmentation=self.segmentation,
            seg_grid_centers=self.seg_grid_centers,
            future_trajs=self.future_trajs,
        )


@dataclass
class VADInferenceOutput:
    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in the ego-frame @ 2Hz"""
    aux_outputs: Optional[VADAuxOutputs] = None
    """aux outputs such as objects, tracks, segmentation and motion forecast"""


class VADRunner:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
        use_col_optim: bool = False,
    ):
        config = Config.fromfile(config_path)
        self.config = config

        self.model: VAD = build_model(
            config.model, train_cfg=None, test_cfg=config.get("test_cfg")
        )

        self.model.eval()
        # load the checkpoint
        if checkpoint_path is not None:
            ckpt = load_checkpoint(self.model, checkpoint_path, map_location="cpu")
            self.classes = ckpt["meta"]["CLASSES"]
        else:
            raise ValueError("checkpoint_path is None")

        # do more stuff here maybe?
        self.model = self.model.to(device)
        self.device = device
        self.preproc_pipeline = Compose(config.inference_pipeline)
        self.use_col_optim = use_col_optim
        self.planning_steps = 6
        self.future_modes = 6
        self.score_treshold = 0.35
        self.occ_filter_range = 5.0  # default parameters taken from UniAD
        self.sigma = 1.0  # default parameters taken from UniAD
        self.alpha_collision = 5.0  # default parameters taken from UniAD
        self._bev_h = self.config.bev_h_
        self._bev_w = self.config.bev_w_
        self._bev_range = (-51.2, 51.2)
        self._bev_resolution = 51.2 * 2 / self._bev_h
        self.counter = 0

        self.reset()

    def reset(self):
        # making a new scene token for each new scene. these are used in the model.
        self.scene_token = str(uuid.uuid4())
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def _preproc_canbus(self, input: VADInferenceInput):
        """Preprocesses the raw canbus signals from nuscenes."""
        rotation = Quaternion(input.can_bus_signals[3:7])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        # extend the canbus signals with the patch angle, first in radians then in degrees
        input.can_bus_signals = np.append(
            input.can_bus_signals, patch_angle / 180 * np.pi
        )
        input.can_bus_signals = np.append(input.can_bus_signals, patch_angle)
        input.can_bus_signals[3:7] = -rotation

    def _occ_mask_from_objects(
        self,
        objects: LiDARInstance3DBoxes,
        future_trajs: torch.Tensor,
    ) -> torch.Tensor:
        occ_mask = np.zeros(
            (1, self.planning_steps, 1, self._bev_h, self._bev_w), dtype=np.int32
        )

        if len(objects.tensor) == 0:
            return torch.from_numpy(occ_mask)
        # add the current position of the objects to the future trajectory n x 6 x 7 x 2
        future_coords = torch.cat(
            [torch.zeros(size=(future_trajs.shape[0], 6, 1, 2)), future_trajs], dim=-2
        )
        # compute the yaw angle of the objects (n x 6 x 6)
        coord_diff = torch.diff(future_coords, dim=-2)
        yaw = torch.atan2(coord_diff[..., 1], coord_diff[..., 0])
        widths = objects.tensor[:, 3].unsqueeze(-1).unsqueeze(-1)
        lengths = objects.tensor[:, 4].unsqueeze(-1).unsqueeze(-1)
        # compute the corners of the objects
        corners = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(yaw) * lengths / 2 + torch.sin(yaw) * widths / 2,
                        torch.sin(yaw) * lengths / 2 - torch.cos(yaw) * widths / 2,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        torch.cos(yaw) * lengths / 2 - torch.sin(yaw) * widths / 2,
                        torch.sin(yaw) * lengths / 2 + torch.cos(yaw) * widths / 2,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -torch.cos(yaw) * lengths / 2 - torch.sin(yaw) * widths / 2,
                        -torch.sin(yaw) * lengths / 2 + torch.cos(yaw) * widths / 2,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -torch.cos(yaw) * lengths / 2 + torch.sin(yaw) * widths / 2,
                        -torch.sin(yaw) * lengths / 2 - torch.cos(yaw) * widths / 2,
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        # add the object position to the corners
        corners += (
            objects.bev[:, :2].unsqueeze(1).unsqueeze(1) + future_trajs
        ).unsqueeze(-2)
        # compute the mask
        corners -= self._bev_range[0] + self._bev_resolution / 2
        corners = torch.round(corners / self._bev_resolution).long().cpu().numpy()

        for i_obj in range(len(corners)):
            for i_mode in range(self.future_modes):
                for i_future in range(self.planning_steps):
                    cv2.fillPoly(
                        occ_mask[0, i_future, 0],
                        [corners[i_obj, i_mode, i_future]],
                        1,
                    )

        return torch.from_numpy(occ_mask)

    def _collision_optimization(
        self, sdc_traj_all: torch.Tensor, occ_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimize SDC trajectory with occupancy instance mask.
        Code adapted from UniAD: https://github.com/OpenDriveLab/UniAD

        Args:
            sdc_traj_all (torch.Tensor): SDC trajectory tensor. shape: 1 x future_ts x 2
            occ_mask (torch.Tensor): Occupancy flow instance mask. shape: 1 x futre_ts x 1 x bev_h x bev_w
        Returns:
            torch.Tensor: Optimized SDC trajectory tensor. shape: 1 x future_ts x 2
        """
        pos_xy_t = []
        valid_occupancy_num = 0

        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        for t in range(self.planning_steps):
            cur_t = min(t + 1, occ_horizon - 1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self._bev_h // 2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self._bev_w // 2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = (
                torch.sum(
                    (sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2]) ** 2, axis=-1
                )
                < self.occ_filter_range**2
            )
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index > 0)
        if valid_occupancy_num == 0:
            return sdc_traj_all

        col_optimizer = CollisionNonlinearOptimizer(
            self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t
        )
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack(
            [sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)],
            axis=-1,
        )
        return torch.tensor(
            sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype
        )

    def preproc(self, input: VADInferenceInput):
        """Preprocess the input data."""
        self._preproc_canbus(input)
        # TODO: make torch version of the preproc (for images) pipeline instead of using mmcv version'

    @torch.no_grad()
    def forward_inference(self, input: VADInferenceInput) -> VADInferenceOutput:
        """Run inference without all the preprocessed dataset stuff."""
        # permute rgb -> bgr
        imgs = input.imgs[:, :, :, ::-1]
        # input to preproc shoudl be dict(img=imgs) where imgs: n x h x w x c in bgr format
        preproc_input = dict(img=imgs, lidar2img=input.lidar2img)
        # run it through the inference pipeline (which is same as eval pipeline except not loading annotations)
        preproc_output = self.preproc_pipeline(preproc_input)
        # collect in array as will convert to tensor, but currently it is a list of arrays (n, h, w, c)
        imgs = np.array(preproc_output["img"])
        # move back to the nchw format
        imgs = np.moveaxis(imgs, -1, 1)
        # convert to tensor and move to device
        imgs = torch.from_numpy(imgs).to(self.device)
        # img should be (1, n, 3, h, w)
        imgs = imgs.unsqueeze(0)
        # we are preproccessing the canbus signals only currently.
        # TODO: fix preproc to include the image preprocessing as well. this is currently done
        # in mmcv (i.e., numpy) and not torch.
        self.preproc(input)

        # we need to emulate the img_metas here in order to run the model.
        img_metas = [
            {
                "scene_token": self.scene_token,
                "can_bus": input.can_bus_signals,
                "lidar2img": preproc_output["lidar2img"],  # lidar2cam @ camera2img
                "img_shape": preproc_output["img_shape"],
                "box_type_3d": LiDARInstance3DBoxes,
            }
        ]

        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0]["can_bus"][:3] = 0
            img_metas[0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]

        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["scene_token"] = self.scene_token
        affix = f"{self.seq}_{self.counter}.pth"
        img_feats = self.model.extract_feat(img=imgs, img_metas=img_metas)
        outs = self.model.pts_bbox_head(
            img_feats,
            img_metas,
            prev_bev=self.prev_frame_info["prev_bev"],
            ego_his_trajs=None,  #  dont have to because the dont use the history trajs
            ego_lcf_feat=None,  # these are not used either
            affix = affix
        )
#####################################################################################################
        planning_loss = None #TODO
        loss_path = LOSS_PATH + '/' + affix
        save_dict = {loss_path:planning_loss}
        thread = threading.Thread(target = save_tensor, args = [save_dict])
        thread.start()
#####################################################################################################

        self.prev_frame_info["prev_bev"] = outs["bev_embed"]

        bbox_list = self.model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)

        bbox_results = []
        for i, (
            bboxes,
            scores,
            labels,
            trajs,
            map_bboxes,
            map_scores,
            map_labels,
            map_pts,
        ) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result["trajs_3d"] = trajs.cpu()
            map_bbox_result = self.model.map_pred2result(
                map_bboxes, map_scores, map_labels, map_pts
            )
            bbox_result.update(map_bbox_result)
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i].cpu()
            bbox_results.append(bbox_result)

        # note that they work with deltas
        trajectory = (
            bbox_results[0]["ego_fut_preds"][input.command].cumsum(dim=-2).numpy()
        )
        future_trajs = (
            bbox_results[0]["trajs_3d"].reshape(-1, 6, 6, 2).cumsum(dim=-2)
        )  # + bboxes.bev[:, :2].unsqueeze(1).unsqueeze(1)
        occ_mask = None
        seg_grid_centers = None
        if self.use_col_optim:
            classes = np.array([self.classes[i] for i in bbox_results[0]["labels_3d"]])
            cls_mask = torch.from_numpy(np.isin(classes, COLLISION_CLASSES_TO_CHECK))
            scores_mask = bbox_results[0]["scores_3d"] >= self.score_treshold
            mask = torch.logical_and(cls_mask, scores_mask)
            # if we dont have any objects to check for collision, we can skip the optimization
            if mask.any():
                occ_mask = self._occ_mask_from_objects(
                    bbox_results[0]["boxes_3d"][mask],
                    future_trajs[mask],
                )
                tmpx = torch.linspace(
                    self._bev_range[0], self._bev_range[1], self._bev_w
                )
                tmpy = torch.linspace(
                    self._bev_range[0], self._bev_range[1], self._bev_h
                )
                tmp_m, tmp_n = torch.meshgrid(tmpx, tmpy)  # indexing 'ij'
                tmp_m, tmp_n = tmp_m.T, tmp_n.T  # change it to the 'xy' mode results
                seg_grid_centers = torch.stack([tmp_m, tmp_n], dim=2).tolist()
                # they only use the first 4 during collision optimization (soley follwing UniAD implementation)
                occ_mask_ = occ_mask[:, :4]
                # but they have the currnent occupancy first, however it is never used so we can just set it to 0
                occ_mask_ = torch.cat(
                    [torch.zeros_like(occ_mask_[:, 0]).unsqueeze(0), occ_mask_], dim=1
                )
                trajectory = (
                    self._collision_optimization(
                        torch.from_numpy(trajectory).unsqueeze(0), occ_mask_
                    )
                    .squeeze(0)
                    .numpy()
                )

        score_mask = bbox_results[0]["scores_3d"] >= self.score_treshold

        if not score_mask.any():
            objects_in_bev = np.empty((0, 5))
            object_scores = np.empty((0))
            object_classes = []
            future_trajs = np.empty((0, 6, 6, 2))
        else:
            objects_in_bev = bbox_results[0]["boxes_3d"].bev[score_mask].cpu().numpy()
            object_scores = bbox_results[0]["scores_3d"][score_mask].cpu().numpy()
            object_classes = [
                self.classes[i] for i in bbox_results[0]["labels_3d"][score_mask]
            ]
            future_trajs = future_trajs[score_mask].cpu().numpy()
        self.counter += 1
        return VADInferenceOutput(
            trajectory=trajectory,
            aux_outputs=VADAuxOutputs(
                objects_in_bev=objects_in_bev.tolist(),
                object_scores=object_scores.tolist(),
                object_classes=object_classes,
                segmentation=occ_mask[0, 0, 0].tolist()
                if occ_mask is not None
                else None,  # bev_h, bev_w
                seg_grid_centers=seg_grid_centers,  # bev_h, bev_w, 2 [x, y]
                future_trajs=future_trajs.tolist(),  # N x 6 modes x 6 future_timesteps x 2 (x, y)
            ),
        )


def _get_sample_input(nusc, nusc_can, scene_name, sample) -> VADInferenceInput:
    timestamp = sample["timestamp"]
    # get the cameras for this sample
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    # ego pose via lidar sensor sample data
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sample_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    ego_translation = np.array(ego_pose["translation"])
    ego_rotation_quat = Quaternion(array=ego_pose["rotation"])
    ego2global = np.eye(4)
    ego2global[:3, 3] = ego_translation
    ego2global[:3, :3] = ego_rotation_quat.rotation_matrix

    # get cameras
    camera_tokens = [sample["data"][camera_type] for camera_type in camera_types]
    # sample data for each camera
    camera_sample_data = [nusc.get("sample_data", token) for token in camera_tokens]
    # get the camera calibrations
    camera_calibrations = [
        nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        for cam in camera_sample_data
    ]
    # get the image filepaths
    image_filepaths = [
        nusc.get_sample_data(cam_token)[0] for cam_token in camera_tokens
    ]
    # get the camera instrinsics
    cam_instrinsics = [
        np.array(cam_calib["camera_intrinsic"]) for cam_calib in camera_calibrations
    ]
    # compute the camera2img and camera2ego transformations
    camera2img = []
    cam2global = []
    for i in range(len(camera_types)):
        # camera 2 image
        c2i = np.eye(4)
        c2i[:3, :3] = cam_instrinsics[i]
        camera2img.append(c2i)
        # camera 2 ego
        c2e = np.eye(4)
        c2e[:3, 3] = np.array(camera_calibrations[i]["translation"])
        c2e[:3, :3] = Quaternion(
            array=camera_calibrations[i]["rotation"]
        ).rotation_matrix
        # ego 2 global (for camera time)
        cam_e2g = nusc.get("ego_pose", camera_sample_data[i]["ego_pose_token"])
        cam_e2g_t = np.array(cam_e2g["translation"])
        cam_e2g_r = Quaternion(array=cam_e2g["rotation"])
        e2g = np.eye(4)
        e2g[:3, 3] = cam_e2g_t
        e2g[:3, :3] = cam_e2g_r.rotation_matrix
        # cam 2 global
        cam2global.append(e2g @ c2e)

    # load the images in rgb hwc format
    images = []
    for filepath in image_filepaths:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    images = np.array(images)

    # get the lidar calibration
    lidar_sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_calibration = nusc.get(
        "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
    )
    lidar_translation = np.array(lidar_calibration["translation"])
    lidar_rotation_quat = Quaternion(array=lidar_calibration["rotation"])
    lidar2ego = np.eye(4)
    lidar2ego[:3, 3] = lidar_translation
    lidar2ego[:3, :3] = lidar_rotation_quat.rotation_matrix

    lidar2global = ego2global @ lidar2ego
    global2lidar = np.linalg.inv(lidar2global)
    # the lidar2img should take into consideration that the the timestamps are not the same
    # because of this we go img -> cam -> ego -> global -> ego' -> lidar
    cam2lidar = [global2lidar.copy() @ c2g for c2g in cam2global]
    lidar2cam = [np.linalg.inv(c2l) for c2l in cam2lidar]
    lidar2img = [c2i @ l2c for c2i, l2c in zip(camera2img, lidar2cam)]

    # get the canbus signals
    pose_messages = nusc_can.get_messages(scene_name, "pose")
    can_times = [pose["utime"] for pose in pose_messages]
    assert np.all(np.diff(can_times) > 0), "canbus times not sorted"
    # find the pose that is less than the current timestamp and closest to it
    pose_idx = np.searchsorted(can_times, timestamp)
    # get the canbus signals in the correct order
    canbus_singal_order = ["pos", "orientation", "accel", "rotation_rate", "vel"]
    canbus_signals = np.concatenate(
        [
            np.array(pose_messages[pose_idx][signal_type])
            for signal_type in canbus_singal_order
        ]
    )

    return VADInferenceInput(
        imgs=images,
        lidar_pose=lidar2global,
        lidar2img=lidar2img,
        timestamp=timestamp,
        can_bus_signals=canbus_signals,
        command=0,  # right
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runner = VADRunner(
        config_path="/VAD/projects/configs/VAD/VAD_inference.py",
        checkpoint_path="/VAD/ckpts/VAD_base.pth",
        device=torch.device(device),
    )

    # only load this for testing
    import cv2
    import matplotlib.pyplot as plt
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes

    # load the first surround-cam in nusc mini
    nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes")
    nusc_can = NuScenesCanBus(dataroot="./data/nuscenes")
    scene_name = "scene-0103"
    scene = [s for s in nusc.scene if s["name"] == scene_name][0]
    # get the first sample in the scene
    sample = nusc.get("sample", scene["first_sample_token"])

    for i in range(60):
        inference_input = _get_sample_input(nusc, nusc_can, scene_name, sample)
        if i > 4:
            inference_input.command = 2  # straight
        plan = runner.forward_inference(inference_input)
        # plot in bev
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(inference_input.imgs[0])
        ax[0].axis("off")

        # save fig
        fig.savefig(f"{scene_name}_{str(i).zfill(3)}_{sample['timestamp']}.png")
        plt.close(fig)
        if sample["next"] == "":
            break
        sample = nusc.get("sample", sample["next"])
