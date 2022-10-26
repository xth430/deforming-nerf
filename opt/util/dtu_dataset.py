# DTU dataset loader
from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np
from scipy.interpolate import CubicSpline
import os


class DTUDataset(DatasetBase):
    """
    DTU dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,
        factor: int = 1,
        scale : Optional[float] = None,
        permutation: bool = True,
        white_bkgd: bool = True,
        n_images = None,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 2/3
            # scene_scale = 0.33
            # scene_scale = 1.0
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size

        split_name = split if split != "test_train" else "train"
        print("LOAD DATA", root)
        print('split:', split)

        imgs, poses, render_poses, [H, W, focal], K, i_split = load_dtu_data(root, half_res=True)

        # NOTE: test 22/3/11
        print('scene_scale:', scene_scale)
        poses[:, :3, 3] *= scene_scale
        render_poses[:, :3, 3] *= scene_scale

        if split == "train":
            self.gt = torch.from_numpy(imgs.astype(np.float32))
            self.c2w = torch.from_numpy(poses.astype(np.float32))
        else:
            self.gt = torch.from_numpy(imgs.astype(np.float32))[:len(render_poses)]
            self.c2w = render_poses
            self.render_c2w = render_poses

        # # debug, temp
        # self.gt = torch.from_numpy(imgs.astype(np.float32))
        # self.c2w = torch.from_numpy(poses.astype(np.float32))

        # print('self.gt.shape:', self.gt.shape)
        # print('self.c2w.shape:', self.c2w.shape)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.intrins_full: Intrin = Intrin(fx, fy, cx, cy)

        # self.n_images, self.h_full, self.w_full, _ = imgs.shape

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        # # Choose a subset of training images
        # if n_images is not None:
        #     if n_images > self.n_images:
        #         print(f'using {self.n_images} available training views instead of the requested {n_images}.')
        #         n_images = self.n_images
        #     self.n_images = n_images
        #     self.gt = self.gt[0:n_images,...]
        #     self.c2w = self.c2w[0:n_images,...]

        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

        self.dataset_type = 'dtu'


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_K_Rt_from_P(P):

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


''' 
    borrowed from pixelnerf:
    https://github.com/sxyu/pixel-nerf/blob/master/eval/gen_video.py#L120
'''
def generate_camera_trajectory():
    print("Using DTU camera trajectory")
    # Use hard-coded pose interpolation from IDR for DTU


    # TODO: test
    # t_in = np.array([0, 0.8, 2.8, 4]).astype(np.float32)
    t_in = np.array([0, 0.8, 3.0, 4]).astype(np.float32)
    pose_quat = torch.tensor(
        [
            [0.7020, 0.1578, -0.4525, -0.5268],
            [0.6766, 0.3176, -0.5179, -0.4161],
            # [0.9085, 0.4020, 0.1139, -0.0025],
            # [0.9698, 0.2121, 0.1203, -0.0039],
            # [0.6766, 0.3176, 0.5179, 0.4161],
            [0.8498,  0.2527,  -0.3154,  -0.3383],
            [0.7020, 0.1578, -0.4525, -0.5268],
        ]
    )
    n_inter = 15
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([2.2, 2.2, 2.2, 2.2]).astype(np.float32) # custom
    # TODO: test


    # t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    # pose_quat = np.array(
    #     [
    #         [0.9698, 0.2121, -0.1203, 0.0039],
    #         [0.7020, 0.1578, -0.4525, -0.5268],
    #         [0.6766, 0.3176, -0.5179, -0.4161],
    #         [0.9085, 0.4020, -0.1139, 0.0025],
    #         [0.9698, 0.2121, -0.1203, 0.0039],
    #     ]
    # )
    # n_inter = 10
    # t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    # scales = np.array([2.2, 2.2, 2.2, 2.2, 2.2]).astype(np.float32) # custom
    # # scales = np.array([4.2, 4.2, 3.8, 3.8, 4.2]).astype(np.float32) # idr


    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat, bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    q_new = torch.from_numpy(q_new).float()

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        new_q = new_q[None, ...]
        R = quat_to_rot(new_q)
        t = R[:, :, 2] * scale
        new_pose = torch.eye(4, dtype=torch.float32)[None, ...]
        new_pose[:, :3, :3] = R
        new_pose[:, :3, 3] = -t

        render_poses.append(new_pose)
    render_poses = torch.cat(render_poses, dim=0)




    # # t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    # # pose_quat = torch.tensor(
    # #     [
    # #         [0.9698, 0.2121, 0.1203, -0.0039],
    # #         [0.7020, 0.1578, 0.4525, 0.5268],
    # #         [0.6766, 0.3176, 0.5179, 0.4161],
    # #         [0.9085, 0.4020, 0.1139, -0.0025],
    # #         [0.9698, 0.2121, 0.1203, -0.0039],
    # #     ]
    # # )
    # # num_views = 40
    # # n_inter = num_views // 5
    # # num_views = n_inter * 5
    # # t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    # # # scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32) # used in pixelnerf
    # # # scales = np.array([3.0, 3.0, 3.0, 3.0, 3.0]).astype(np.float32) # original idr paper
    # # # scales = np.array([3.5, 3.5, 3.5, 3.5, 3.5]).astype(np.float32) # custom
    # # scales = np.array([2.2, 2.2, 2.2, 2.2, 2.2]).astype(np.float32) # custom

    # s_new = CubicSpline(t_in, scales, bc_type="periodic")
    # s_new = s_new(t_out)

    # q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
    # q_new = q_new(t_out)
    # q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    # q_new = torch.from_numpy(q_new).float()

    # # print(q_new)

    # render_poses = []
    # for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
    #     new_q = new_q.unsqueeze(0)
    #     R = quat_to_rot(new_q)
    #     t = R[:, :, 2] * scale
    #     new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    #     new_pose[:, :3, :3] = R
    #     new_pose[:, :3, 3] = -t
        
    #     # NOTE: check here, 2/23
    #     new_pose[:, [0], :] *= -1
    #     new_pose[:, :, [0]] *= -1

    #     render_poses.append(new_pose)
    # render_poses = torch.cat(render_poses, dim=0)
    
    return render_poses


def load_dtu_data(basedir, half_res=False, testskip=1):

    imgs_all = []
    msks_all = []

    img_dir = basedir + '/image'
    img_paths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir)])
    # msk_dir = basedir + '/mask'
    # msk_paths = sorted([os.path.join(msk_dir, fname) for fname in os.listdir(msk_dir)])
    
    n_imgs = len(img_paths)

    cam_file = basedir + '/cameras.npz'

    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_imgs)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_imgs)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):

        # orig
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(P)
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)

    for img_path in img_paths:
        imgs_all.append(imageio.imread(img_path))

    # for msk_path in msk_paths:
    #     msks_all.append(imageio.imread(msk_path, as_gray=True))

    imgs = (np.array(imgs_all) / 255.).astype(np.float32)


    # # msks = (np.array(msks_all) / 255.).astype(np.float32)
    # msks = (np.array(msks_all) > 127.5)[..., None]

    # # use mask
    # imgs = imgs * msks

    poses = np.array(pose_all).astype(np.float32)
    intrinsics = np.array(intrinsics_all).astype(np.float32)

    # focal = intrinsics[0][0, 0]
    focal = 0
    
    H, W = imgs[0].shape[:2]

    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,48+1)[:-1]], 0)
    render_poses = generate_camera_trajectory()

    if half_res:

        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    # intrics
    K = np.eye(3)
    K[:2] = np.mean(intrinsics, axis=0)[:2, :3] / 2
    
    num_test = 5
    i_split = [np.arange(0, len(imgs)-num_test), np.arange(len(imgs)-num_test, len(imgs)-1), [63]]

    return imgs, poses, render_poses, [H, W, focal], K, i_split