import os, trimesh, mcubes, argparse, warnings
import numpy as np
from trimesh.voxel import creation
from util import smoothing


parser = argparse.ArgumentParser()
parser.add_argument('mesh_path', type=str)
# grid reso and symmetry
parser.add_argument('--radius', default=4, type=int, help='grid resolution=(m,m,m), where m=2*radius+1, which controls the roughness of the cage')
parser.add_argument('--sym_axis', default='', type=str, help='specify the axis of symmetry, e.g., "x", "z", "yz", ...')
# cage smoothing
parser.add_argument('--smoothing', default=True, type=bool, help='apply smoothing to the cage')
parser.add_argument('--num_iter', default=10, type=int, help='# steps of cage smoothing')
parser.add_argument('--lbd', default=0.2, type=float, help='lambda controls the speed of cage smoothing')
parser.add_argument('--use_improved', default=True, type=bool, help='using improved cage smoothing instead of laplacian')
# post-processing
parser.add_argument('--shrink_factor', default=1.0, type=float, help='shrink the generated cage. usually 0.85~0.95 in practice')

args = parser.parse_args()
export_path = os.path.join(os.path.dirname(args.mesh_path), 'cage.obj')

print('[INFO] loading mesh from:', args.mesh_path)
mesh = trimesh.load(args.mesh_path)
v, t = mesh.vertices, mesh.faces
print('[INFO] fine mesh has {:d} vertices and {:d} faces'.format(len(v), len(t)))

# voxelize mesh to grid
print('[INFO] voxelizing mesh... (this may take a while)')
point = [0, 0, 0]
radius, pitch = args.radius, 1. / (args.radius + 0.5)
voxel = trimesh.voxel.creation.local_voxelize(mesh, point, pitch=pitch, radius=radius, fill=True)
densitygrid = voxel.encoding.dense
print('[DONE] voxelization done! voxel size:', voxel.shape)

# symmetry
if 'x' in args.sym_axis:
    densitygrid = np.logical_or(densitygrid, densitygrid[::-1, :, :])
if 'y' in args.sym_axis:
    densitygrid = np.logical_or(densitygrid, densitygrid[:, ::-1, :])
if 'z' in args.sym_axis:
    densitygrid = np.logical_or(densitygrid, densitygrid[:, :, ::-1])

# marching cubes
v, t = mcubes.marching_cubes(densitygrid, 0.5)

# grid2world
v = v / radius - 1

# shrink
if args.shrink_factor < 1.0:
    v *= args.shrink_factor

# cage smoothing algorithm from: http://www.cad.zju.edu.cn/home/hwlin/pdf_files/Automatic-generation-of-coarse-bounding-cages-from-dense-meshes.pdf
cage = trimesh.Trimesh(v, t)
if args.smoothing:
    cage = smoothing(cage, num_iter=args.num_iter, lbd=args.lbd, use_improved=args.use_improved)
v, t = cage.vertices, cage.faces

# cage should be a watertight mesh
if not cage.is_watertight:
    warnings.warn('Cage is not watertight!', UserWarning)

# export mesh
mcubes.export_obj(v, t, export_path)
print('[INFO] generated cage has {:d} vertices and {:d} faces'.format(len(v), len(t)))
print('[DONE] OBJ file saved to:', export_path)
