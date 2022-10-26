import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util import config_util

import imageio
from tqdm import tqdm

from deformation.deformation import CBD, volume_render_image_deformed
from deformation.util import load_cages, get_near_far, visualize_opt

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--no_imsave',
                    action='store_true',
                    default=False,
                    help="Disable image saving (can still save video; MUCH faster)")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

# Cage setting
parser.add_argument('--cage_source', type=str, default='cage.obj', help='.obj file of source cage')
parser.add_argument('--cage_target', type=str, default='cage_deformed.obj', help='.obj file of target cage')

parser.add_argument('--coord_type', type=str, default='HC', help='type of the cage coordinate')
parser.add_argument('--coord_reso', type=int, default=128, help='resolution of the cage coordinate')
parser.add_argument('--deform_viewdirs', default=True, help='use viewdirs deformation')

# Render and interpolate setting
parser.add_argument('--render_orig', action='store_true', default=False, help="rendering original scene imgs and depths for comparison")
parser.add_argument('--cam_id', type=int, default=-1, help='fixed camera id')
parser.add_argument('--interpolate', action='store_true', default=False, help="cage interpolation")
parser.add_argument('--num_interpolate_frame', type=int, default=60, help='num of interpolated frame')
parser.add_argument('--loop', action='store_true', default=False, help="render loop animation")


args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'

# Load cages
cage_source, cage_target = load_cages(args)

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_dir = path.join(path.dirname(args.ckpt),
            'train_renders' if args.train else 'test_renders')

if args.render_orig:
    render_dir += '_orig'
    cage_target = cage_source # only used for cage visualization
else:
    render_dir += f'_deformed_{args.coord_type}'

if args.interpolate:
    render_dir += '_interpolate'

if args.cam_id >= 0:
    render_dir += f'_cam{args.cam_id:04d}'

dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))

near, far = get_near_far(dset)

grid = svox2.SparseGrid.load(args.ckpt, device=device)
grid.white_bkgd = args.white_bkgd

if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_data.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        render_dir += '_nofg'

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_dir += '_blackbg'
    grid.opt.background_brightness = 0.0

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

if not args.no_imsave:
    print('Will write out all frames as PNG (this take most of the time)')


# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    if args.cam_id >= 0:
        n_images = args.num_interpolate_frame if args.interpolate else 1
    img_eval_interval = max(n_images // args.n_eval, 1)

    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)

    frames = []

    # deformation class
    cbd = CBD(cage_source, cage_target, coord_type=args.coord_type, res=args.coord_reso, deform_viewdirs=args.deform_viewdirs)

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        # choose camera
        cam_id = args.cam_id if args.cam_id >= 0 else img_id

        cam = svox2.Camera(c2ws[cam_id],
                           dset.intrins.get('fx', cam_id),
                           dset.intrins.get('fy', cam_id),
                           dset.intrins.get('cx', cam_id) + (w - dset_w) * 0.5,
                           dset.intrins.get('cy', cam_id) + (h - dset_h) * 0.5,
                           w, h,
                           ndc_coeffs=dset.ndc_coeffs)

        cam.near, cam.far = near, far

        # interpolate
        if args.interpolate:
            step = img_id
            if (dset.dataset_type in ['nerf', 'dtu']) and args.loop:
                step = (n_images - abs(2*img_id-n_images))
            # compute cage interpolation
            cbd.interpolate_cage(step=step, max_steps=n_images, cache=False)

        # render 
        im, disp = volume_render_image_deformed(grid, cam, cbd, render_orig=args.render_orig)
        im.clamp_(0.0, 1.0)

        # add disparity map and cages for better visualization
        im = visualize_opt(im, disp, cam, cbd, dset)

        img_path = path.join(render_dir, f'{img_id:04d}.png');

        im = (im * 255).astype(np.uint8)
        if not args.no_imsave:
            imageio.imwrite(img_path,im)
        if not args.no_vid:
            frames.append(im)
        
        im = None
    
    if not args.no_vid and len(frames) > 1:
        vid_path = render_dir + '.gif'
        imageio.mimwrite(vid_path, frames, fps=args.fps)
        print('video saved to ', vid_path)