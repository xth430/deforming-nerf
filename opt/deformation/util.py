import numpy as np
import trimesh

import torch

import pyrender
from pyrender.constants import RenderFlags
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def load_cages(args):

    cage_path = os.path.join(os.path.dirname(args.ckpt), "cages")
    cage_source = trimesh.load_mesh(os.path.join(cage_path, args.cage_source), process=False)
    cage_target = trimesh.load_mesh(os.path.join(cage_path, args.cage_target), process=False)

    assert cage_source.is_watertight, "Cage (source) is not watertight!"
    assert cage_target.is_watertight, "Cage (target) is not watertight!"

    return cage_source, cage_target


# hard-coded near, far
def get_near_far(dset):

    if dset.dataset_type == 'dtu':
        near, far = 0.9, 3.0 # scene scale: 0.667
    elif dset.dataset_type == 'nerf':
        near, far = 1.6, 3.6
    elif dset.dataset_type == 'nsvf':
        # near, far = 0.6, 1.4
        near, far = 0.4, 1.6 # toad, scene scale: 0.317
    else:
        raise ValueError('Unknown dataset!')

    return near, far


def diagonal_dot(a, b):
    return torch.matmul(a * b, torch.ones(a.shape[1]).to(a.device))

def points_to_barycentric(triangles, points):

    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = points - triangles[:, 0].view((-1, 3))

    dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
    dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
    dot02 = diagonal_dot(edge_vectors[:, 0], w)
    dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
    dot12 = diagonal_dot(edge_vectors[:, 1], w)

    inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)

    barycentric = torch.zeros(len(triangles), 3).to(points.device)
    barycentric[:, 2] = (dot00 * dot12 - dot01 *
                         dot02) * inverse_denominator
    barycentric[:, 1] = (dot11 * dot02 - dot01 *
                         dot12) * inverse_denominator
    barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
    return barycentric


def barycentric_to_points(triangles, barycentric):
    return (triangles * barycentric.view((-1, 3, 1))).sum(dim=1)


def mesh_to_voxel(mesh, res=128):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        mesh (Trimesh): [num_rays, num_samples along ray]. Trimesh object. 
    Returns:
        voxel_trimesh (Trimesh.Voxel): [num_points, 3]. Voxel Representation.
    """
    verts = mesh.vertices
    max_dhw, min_dhw = np.max(verts, 0), np.min(verts, 0)
    pitch = np.max(max_dhw - min_dhw) / (res - 2)
    voxel_trimesh = mesh.voxelized(pitch=pitch, method='subdivide') # here may not a cube (e.g., (127, 127, 128))
    voxel_trimesh.fill()
    return voxel_trimesh


def render_cage(im, cage, cam, alpha=0.5):

    if im.dim() == 2:
        im = im.unsqueeze(-1).repeat(1,1,3)

    cage = pyrender.Mesh.from_trimesh(cage)
    scene = pyrender.Scene()
    scene.add(cage)
    camera = pyrender.camera.IntrinsicsCamera(
        fx=cam.fx,
        fy=cam.fy,
        cx=cam.cx,
        cy=cam.cy,
    )
    pose = cam.c2w.to('cpu').detach().numpy().copy()
    pose[:, [1,2]] *= -1 # opencv-to-opengl
    scene.add(camera, pose=pose)
    r = pyrender.OffscreenRenderer(cam.width, cam.height)
    flags = RenderFlags.ALL_WIREFRAME | RenderFlags.SKIP_CULL_FACES
    color, depth = r.render(scene, flags=flags)

    color = torch.from_numpy(color.astype(np.float32)).clone().to(im) / 255
    depth = torch.from_numpy(depth.astype(np.float32)).clone().to(im)

    rgb = (0, 0, 0)
    rgb = torch.tensor(rgb).to(color)[None, :] / 255

    color -= 1 # [-1, 0]
    color *= (1-rgb)

    cage_mask = (color<0.0).any(-1)
    im[cage_mask] += color[cage_mask]*alpha
    im.clamp_min_(0)

    return im


def visualize_opt(im, disp, cam, cbd, dset, disp_thres=0.36, alpha=0.2):

    # # concat disp map # for nerf-synthetic
    if dset.dataset_type == 'nerf':
        disp_scale = 1.4
        disp = disp * disp_scale

    if dset.dataset_type == 'nsvf':
        # disp_scale = 0.6 # robot
        disp_scale = 0.4 # toad
        disp = disp * disp_scale

    # set background color
    disp[(disp>1000)|(disp<disp_thres)] = disp_thres

    # render cage
    disp = render_cage(disp, cbd.cage_current, cam, alpha=alpha)
    disp.clamp_max_(1.0)

    im = im.cpu().numpy()
    disp = disp.cpu().numpy()
    
    im = np.concatenate([im, disp], axis=1)

    return im
