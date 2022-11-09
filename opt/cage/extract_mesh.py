import os, svox2, mcubes, torch, argparse, trimesh
import torch.nn.functional as F
import numpy as np

from trimesh.util import log
log.disabled = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)
parser.add_argument('--resfactor', default=0.5, type=float)
parser.add_argument('--mc_thres', default=20., type=float, help='threshold of marching cubes')
args = parser.parse_args()

export_path = os.path.join(os.path.dirname(args.ckpt), 'fine.obj')
print('[INFO] loading Plenoxel model from:', args.ckpt)
grid = svox2.SparseGrid.load(args.ckpt, device=device)
resfactor = args.resfactor

# load original density grid
density_data, links = grid.density_data, grid.links
msk = links >= 0
densitygrid_orig = -1e9 * torch.ones_like(links, dtype=torch.float32)  # (640^3)
densitygrid_orig[msk] = density_data[:, 0]

resx = int(grid.shape[0]*resfactor)
resy = int(grid.shape[1]*resfactor)
resz = int(grid.shape[2]*resfactor)

xx, yy, zz = torch.meshgrid(torch.arange(resx), torch.arange(resy), torch.arange(resz))
xx, yy, zz = (xx+0.5)/resx, (yy+0.5)/resy, (zz+0.5)/resz
coords = torch.stack((xx, yy, zz), dim=-1)[None, ...].to(device) # (1, resx, resy, resz, 3)
coords = coords*2 - 1

# sample low-res density grid
densitygrid = F.grid_sample(densitygrid_orig[None, None, ...], coords, align_corners=False)
densitygrid = densitygrid.squeeze(0).squeeze(0).to('cpu').detach().numpy().copy()

# marching cubes
print('[INFO] applying marching cubes...')
v, t = mcubes.marching_cubes(densitygrid, args.mc_thres) #adjust value to your scene. start with 0.

# grid2world
v = v / resfactor
v = grid.grid2world(torch.from_numpy(v.astype(np.float32)).clone().to(device))
v = v.to('cpu').detach().numpy().copy()

# remove noisy meshes
mesh = trimesh.Trimesh(v, t)
mesh_split = mesh.split(only_watertight=False)
if len(mesh_split) > 0:
    print('[INFO] ignore the irrelevant noisy meshes')
    mesh = sorted(mesh_split, key=lambda x:x.vertices.shape[0])[-1]
    v, t = mesh.vertices, mesh.faces
print('[INFO] fine mesh has {:d} vertices and {:d} faces'.format(len(v), len(t)))

# export mesh
mcubes.export_obj(v, t, export_path)
print('[DONE] mesh saved to:', export_path)
