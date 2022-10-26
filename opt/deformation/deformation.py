import torch
import svox2
import svox2.utils
import numpy as np

import numpy as np
import trimesh
import time

import torch
import torch.nn.functional as F

import copy

from .util import points_to_barycentric, barycentric_to_points, mesh_to_voxel
from .coords_util import mean_value_coordinates_3D, green_coordinates_3D, compute_face_normals_and_areas


def rays_to_points(rays, num_samples=1024, near=0.9, far=3.0):  # (dtu scan83)
    """Plenoxels rays to points in world coords.
    Args:
        rays: Plenoxels rays
    Returns:
        points: [num_rays, num_samples along ray, 3]. World coordinates of sampling points of rays.
    """
    origins, dirs = rays.origins, rays.dirs
    t = torch.linspace(0, 1, steps=num_samples).to(origins.device)[None, :]  # (1, N_sam)
    z_vals = near + (far - near) * t
    points = origins.unsqueeze(1) + z_vals[..., None] * dirs.unsqueeze(1)

    return points, z_vals


def dirs_to_shmul(dirs):
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-6) # [B, 3]
    shmul = svox2.utils.eval_sh_bases(9, dirs)
    return shmul


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).to(dists).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]

    rgb = raw[..., :3]  # [N_rays, N_samples, 3]

    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # NOTE: weights for adding bg
    weights_bg = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists), 1.0 - alpha + 1e-10], -1),-1,)
    weights = alpha * weights_bg[:, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./ torch.max(1e-10 * torch.ones_like(depth_map).to(dists), depth_map / (torch.sum(weights, -1) + 1e-10))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, weights_bg


def mask_outside_sphere(raw, points, radius=1.0):
    """Mask color & density for points out of radius.
    Args:
        raw: [B, N, 4]. Prediction from model.
        points: [B, N, 3]. Points in canonical space.
        radius: float. Radius of the sphere.
    Returns:
        raw: [B, N, 4]. Masked raw output.
    """
    mask = (torch.norm(points, dim=-1) > radius)
    raw[mask] = 0.0

    return raw


def volume_render_image_deformed(grid, cam, cbd, render_orig=False, batch_size=40000, num_samples=512):

    # Manually generate rays for now
    rays = cam.gen_rays()

    rgb, acc, disp, bgweights = [], [], [], []

    for batch_start in range(0, cam.height * cam.width, batch_size):

        batched_rays = rays[batch_start : batch_start + batch_size]
        points, z_vals = rays_to_points(batched_rays, num_samples=num_samples, near=cam.near, far=cam.far)  # (B, N_sam, 3)

        B, _, _ = points.size()
        points = points.view(-1, 3)

        viewdirs = batched_rays.dirs

        # cage-based deformation
        if not render_orig:
            points, viewdirs = cbd.deformed_to_canonical(points, viewdirs)

        density, color = grid.sample(points)
        density, color = density.view(-1, num_samples, 1), color.view(-1, num_samples, 3, grid.basis_dim)  # (B, N_sam, 1), (B, N_sam, 3, 9)
        points = points.view(-1, num_samples, 3) # [B, N_sam, 3]

        # Spherical harmonic
        # dirs: [B, 3] (original viewdirs) or [B*N, 3] (deformed viewdirs)
        shmul = dirs_to_shmul(viewdirs)
        shmul = shmul.view(B, -1, 1, grid.basis_dim) # [B, 1, 1, 9] or [B, N_sam, 1, 9]
        color = torch.clamp_min(torch.sum(shmul*color, dim=-1) + 0.5, 0.0)  # [B', N_sam, 3]

        # volume rendering
        raw = torch.cat([color, density], dim=-1)
        z_vals = z_vals.expand((color.shape[0], -1))

        # NOTE: check here, remove points that outside a unit sphere
        raw = mask_outside_sphere(raw, points, radius=1.0)

        rgb_map, disp_map, acc_map, _, _, weights_bg = raw2outputs(raw, z_vals, batched_rays.dirs, white_bkgd=grid.white_bkgd)

        rgb.append(rgb_map)
        acc.append(acc_map)
        disp.append(disp_map)
        bgweights.append(weights_bg[:, -1])

    im = torch.cat(rgb, dim=0).view(cam.height, cam.width, -1)
    acc = torch.cat(acc, dim=0).view(cam.height, cam.width)
    disp = torch.cat(disp, dim=0).view(cam.height, cam.width)
    bgweights = torch.cat(bgweights, dim=0).view(cam.height, cam.width, 1)

    # background rendering
    if grid.background_nlayers > 0:

        # background grid
        grid_bg = copy.deepcopy(grid)
        grid_bg.density_data.data[:] = 0.0 # remove foreground

        bg = grid_bg.volume_render_image(cam, use_kernel=True, return_raylen=False)
        im = im + bg*bgweights

        # [tmp] remove noisy "cloud", ONLY FOR DTU dataset
        thres = 0.4
        disp[acc<thres] = 0.0

    return im, disp


'''
    Cage-based deformation class
'''
class CBD:

    def __init__(self, cage_source, cage_target, coord_type='HC', res=128, deform_viewdirs=True):
        self.coord_type = coord_type
        self.res = res
        self.deform_viewdirs = deform_viewdirs
        self.set_cage_source(cage_source)
        self.set_cage_target(cage_target)
        self.cage_coordinate = None
        self.interpolate_cache = {}  

    def set_cage_source(self, cage_source):
        self.cage_source = cage_source
        self.cage_source_voxel = VoxelBase(cage_source)

    def set_cage_target(self, cage_target):
        self.cage_target = cage_target
        self.cage_current = cage_target

    def deformed_to_canonical(self, points, viewdirs):
        
        if not self.cage_coordinate:
            self.cage_coordinate = compute_cage_coordinate(self.cage_target, self.cage_source, coord_type=self.coord_type, res=self.res)
        
        # deform points
        points = deform_with_cage(self.cage_coordinate, self.cage_source_voxel, points, coord_type=self.coord_type)
        
        # deform viewdirs
        if self.deform_viewdirs:
            viewdirs = self.deform_viewdir(points, viewdirs)
        
        return points, viewdirs

    def canonical_to_deformed(self, points, viewdirs):
        pass

    def interpolate_cage(self, step=0, max_steps=60, cache=False):
        if step not in self.interpolate_cache:
            vertices_interpolate = (step / max_steps) * self.cage_target.vertices + ((max_steps - step) / max_steps) * self.cage_source.vertices
            cage_interpolate = trimesh.Trimesh(vertices=vertices_interpolate, faces=self.cage_source.faces)
            self.cage_current = cage_interpolate
            self.cage_coordinate = compute_cage_coordinate(cage_interpolate, self.cage_source, coord_type=self.coord_type, res=self.res)
            if cache:
                self.interpolate_cache[step] = (self.cage_coordinate, self.cage_current)
        else:
            self.cage_coordinate, self.cage_current = self.interpolate_cache[step]

    def deform_viewdir(self, points, viewdirs):
        """Mask color & density for points out of radius.
        Args:
            points: [B*N, 3]. 
            viewdirs: [B, 3]
        Returns:
            dirs: [B*N, 3]. 
        """
        B, _ = viewdirs.size()
        points = points.view(B, -1, 3) # [B, N_sam, 3]

        # calculate dirs using difference approximation
        viewdirs = points[:, 1:, :] - points[:, :-1, :] # [B, N_sam-1, 3]
        viewdirs = torch.cat((viewdirs, viewdirs[:,[-1],:]), dim=1)
        viewdirs = viewdirs.view(-1, 3) # [B*N_sam, 3]

        # remove bad viewdirs
        mask = (torch.norm(viewdirs, dim=-1) < 1e-6) | (torch.norm(viewdirs, dim=-1) > 1e6)
        viewdirs[mask] = 1.0

        return viewdirs


'''
    Voxel base for deformation
'''
class VoxelBase:

    def __init__(self, mesh, res=128) -> None:
        self.mesh = mesh
        self.res = res
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        voxel_trimesh = mesh_to_voxel(mesh, res=res)

        # w2g matrix
        trans = self._to_tensor(voxel_trimesh._transform.matrix)
        trans_inv = torch.linalg.inv(trans)
        self._R, self._T = trans_inv[:3, :3], trans_inv[:3, 3:]

        # padding zero
        pad_width = [(0, self.res-l) for l in voxel_trimesh.matrix.shape]
        voxel_bin = np.pad(voxel_trimesh.matrix.astype(np.float32), pad_width, mode='constant') # pad voxel to a cube with shape (res^3)
        # # using erosion
        # voxel_bin_erosed = ndimage.binary_erosion(voxel_bin, iterations=2).astype(voxel_bin.dtype)
        voxel_bin_erosed = voxel_bin

        self.voxel_bin = self._to_tensor(voxel_bin)[..., None]
        self.voxel_bin_erosed = self._to_tensor(voxel_bin_erosed)[..., None]

    def _to_tensor(self, x):
        return torch.from_numpy(x.astype(np.float32)).clone().to(self.device)
    
    def _to_numpy(self, x):
        return x.to('cpu').detach().numpy().copy()

    def _world2grid(self, points):
        if 'torch' not in str(type(points)):
            points = self._to_tensor(points)
        return torch.round(self._world2grid_interpolate(points)).long()

    def _world2grid_interpolate(self, points):
        return (torch.mm(self._R, points.t()) + self._T).t() # xyz->dhw

    def _grid2world_interpolate(self, points):
        return (torch.mm(torch.linalg.inv(self._R), (points.t() - self._T))).t() # dhw->xyz

    def interpolate(self, points, voxel, mode='bilinear', align_corners=False):
        # points (torch.Tensor): [N, 3] in world coordinates
        # voxel (torch.Tensor): [r,r,r,C]
        voxel = voxel.permute(3,2,1,0).unsqueeze(0) # [1,C,r,r,r]
        grid = 2 * self._world2grid_interpolate(points) / (self.res-(align_corners>0)) - 1
        grid = grid.view(1, 1, 1, -1, 3)
        lerp = F.grid_sample(voxel, grid, mode=mode, align_corners=align_corners) # ncdhw
        lerp = lerp.permute(0,2,3,4,1).squeeze(0).squeeze(0).squeeze(0)
        return lerp


'''
    Harmonic coordinates
'''
class Harmonic(VoxelBase):

    def __init__(self, mesh, res=128, n_hier_stage=4, n_iter_per_stage=300, opt_thres=1e-6) -> None:
        super().__init__(mesh, res)
        self.res = res
        self.n_hier_stage = n_hier_stage
        self.n_iter_per_stage = n_iter_per_stage
        self.opt_thres = opt_thres
        
        self.voxel = torch.zeros((self.res, self.res, self.res, len(self.mesh.vertices))).to(self.device)
        self._init_boundary_conditions()
        self.voxel = self._update_boundary_conditions(self.voxel, self.dhw)

    def _sample_surface_points(self):
        points, face_ids = trimesh.sample.sample_surface_even(self.mesh, count=200000)
        points = self._to_tensor(points)
        dhw = self._world2grid(points)

        triangles = self.mesh.faces[face_ids]
        verts_triangles = self._to_tensor(self.mesh.vertices[triangles])
        barycentric = points_to_barycentric(verts_triangles, points)
        coords_triangles = torch.eye(len(self.mesh.vertices)).to(self.device)[triangles]
        coords = barycentric_to_points(coords_triangles, barycentric)
        return dhw, coords

    def _vertex_points(self):
        dhw = self._world2grid(self.mesh.vertices)
        coords = torch.eye(len(self.mesh.vertices)).to(self.device)
        return dhw, coords

    def _init_boundary_conditions(self):
        dhw_ver, coords_ver = self._vertex_points()
        dhw_sur, coords_sur = self._sample_surface_points()
        dhw, coords = torch.cat((dhw_ver, dhw_sur), dim=0), torch.cat((coords_ver, coords_sur), dim=0)
        self.dhw = dhw
        self.coords = coords

    def _update_boundary_conditions(self, voxel, dhw):
        voxel[dhw[:, 0], dhw[:, 1], dhw[:, 2]] = self.coords
        return voxel

    def _laplacian_smoothing(self, voxel, dhw):
        
        voxel[1:-1, 1:-1, 1:-1] = (
            voxel[2:, 1:-1, 1:-1] + voxel[:-2, 1:-1, 1:-1] + 
            voxel[1:-1:, 2:, 1:-1] + voxel[1:-1, :-2, 1:-1] + 
            voxel[1:-1, 1:-1, 2:] + voxel[1:-1, 1:-1, :-2]
        ) / 6.0

        voxel = self._update_boundary_conditions(voxel, dhw)

        return voxel

    def optimize_voxel_hierarchical(self):

        voxel_hier, dhw_hier = [self.voxel], [self.dhw]

        # init hierarchical voxels
        for _ in range(self.n_hier_stage-1):
            voxel_hier.append(F.max_pool3d(voxel_hier[-1].permute(3,0,1,2).unsqueeze(0), kernel_size=2).squeeze(0).permute(1,2,3,0))
            dhw_hier.append(dhw_hier[-1] // 2)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            print(f'Computing HC... (gird size: {self.res}^3)')
            voxel = voxel_hier[-1]
            for stage in list(range(self.n_hier_stage))[::-1]:
                # print('optimizing stage {} ...'.format(stage))
                for i in range(self.n_iter_per_stage):
                    voxel_pre = voxel.clone()
                    voxel = self._laplacian_smoothing(voxel, dhw_hier[stage])

                    if (torch.mean(torch.abs(voxel_pre - voxel)) < self.opt_thres) or (i == self.n_iter_per_stage-1):
                        # print('N_iter:', i+1)
                        if stage != 0:
                            voxel = F.interpolate(voxel.permute(3,0,1,2).unsqueeze(0), scale_factor=2, mode='trilinear', align_corners=False).squeeze(0).permute(1,2,3,0)
                        break

        self.voxel = voxel * self.voxel_bin_erosed
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print('Done! time: {:.2f} sec.'.format(elapsed_time))

    def points_to_weights(self, points):
        return self.interpolate(points, self.voxel)


'''
    Mean value coordinates
'''
class MeanValue(VoxelBase):

    def __init__(self, mesh, res=128, batch_size=200000) -> None:
        super().__init__(mesh, res)
        
        xx, yy, zz = torch.meshgrid(torch.arange(self.res), torch.arange(self.res), torch.arange(self.res))
        grid_coords = torch.stack((xx,yy,zz), dim=-1).view(-1, 3).to(self.device) # (self.res**3, 3)
        
        faces, vertices = self._to_tensor(self.mesh.faces), self._to_tensor(self.mesh.vertices)
        vertices = self._world2grid_interpolate(vertices)
        faces, vertices = faces.long()[None, ...], vertices[None, ...]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            num_points = len(grid_coords)
            print(f'Computing MVC... (# of grid points: {res}^3={num_points})')
            wjs = []
            for i in range(0, num_points, batch_size):
                query = grid_coords[i:i+batch_size][None, ...]
                wj = mean_value_coordinates_3D(query, vertices, faces)
                # wj, _, _ = green_coordinates_3D(query, vertices, faces) # green coordinates
                wjs.append(wj)
            wj = torch.cat(wjs, 1).to(self.device)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print('Done! time: {:.2f} sec.'.format(elapsed_time))

        self.voxel = wj.view(self.res, self.res, self.res, -1)
        del grid_coords
        # remove points outside the cage
        self.voxel *= self.voxel_bin

    def points_to_weights(self, points):
        return self.interpolate(points, self.voxel)


'''
    Green coordinates
'''
class Green(VoxelBase):

    def __init__(self, mesh, mesh_deformed, res=128, batch_size=200000) -> None:
        super().__init__(mesh, res)
        
        xx, yy, zz = torch.meshgrid(torch.arange(self.res), torch.arange(self.res), torch.arange(self.res))
        grid_coords = torch.stack((xx,yy,zz), dim=-1).view(-1, 3).to(self.device) # (self.res**3, 3)
        
        faces, vertices_w = self._to_tensor(self.mesh.faces), self._to_tensor(self.mesh.vertices)
        vertices = self._world2grid_interpolate(vertices_w)
        faces, vertices = faces.long()[None, ...], vertices[None, ...]
        vertices_w = vertices_w[None, ...]

        vertices_deformed = self._to_tensor(mesh_deformed.vertices)[None, ...]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            num_points = len(grid_coords)
            print(f'Computing GC... (# of grid points: {res}^3={num_points})')
            wjs = []
            add_f = []
            for i in range(0, num_points, batch_size):
                query = grid_coords[i:i+batch_size]
                wj, coords_F, _ = green_coordinates_3D(self._grid2world_interpolate(query)[None, ...], vertices_w, faces) # green coordinates
                cage_deformed_FN, _ = compute_face_normals_and_areas(vertices_deformed, faces)
                wjs.append(wj)
                add_f.append(torch.sum(coords_F.unsqueeze(-1)*cage_deformed_FN.unsqueeze(1), dim=-2))
            wj = torch.cat(wjs, 1).to(self.device)
            add_f = torch.cat(add_f, 1).to(self.device)

        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print('Done! time: {:.2f} sec.'.format(elapsed_time))

        self.voxel = wj.view(self.res, self.res, self.res, -1)
        self.add_f = add_f.view(self.res, self.res, self.res, -1)
        del grid_coords
        # remove points outside the cage
        self.voxel = self.voxel * self.voxel_bin_erosed + 1e9 * (1-self.voxel_bin_erosed)
        self.add_f = self.add_f * self.voxel_bin_erosed + 1e9 * (1-self.voxel_bin_erosed)

    def points_to_weights(self, points):
        return self.interpolate(points, self.voxel)
    
    def points_to_add_f(self, points):
        return self.interpolate(points, self.add_f)



# =============================================================================

def _to_tensor(x, device="cuda:0"):
    return torch.from_numpy(x.astype(np.float32)).clone().to(device)


def _to_numpy(x):
    return x.to("cpu").detach().numpy().copy()


def deform_with_cage(coord, cage_voxel, points, coord_type='HC'):
    deform_dict = {
        'MVC': deform_with_MVC,
        'HC': deform_with_HC,
        'GC': deform_with_GC
    }
    return deform_dict[coord_type](coord, cage_voxel, points)


def compute_cage_coordinate(cage_deformed, cage_init, coord_type='HC', res=128):
    if coord_type == 'HC':
        coord = Harmonic(cage_deformed, res=res)
        coord.optimize_voxel_hierarchical()
    elif coord_type == 'MVC':
        coord = MeanValue(cage_deformed, res=res)
    elif coord_type == 'GC':
        coord = Green(cage_deformed, cage_init, res=res)
    else:
        raise ValueError('Wrong coordinates type!')
    return coord


def deform_with_HC(hc, cage_voxel, points):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        hc (Harmonic): [num_rays, num_samples along ray, 4]. Optimized harmonic coordinates.
        cage (Trimesh): [num_rays, num_samples along ray]. Trimesh object.
        points (torch.Tensor): [num_points, 3]. Sampling points.
    Returns:
        deformed: [num_points, 3]. Deformed sampling points.
    """

    cage_verts = cage_voxel.mesh.vertices
    weights = hc.points_to_weights(points)  # (N,C)

    is_inside = weights.sum(-1) > 0.98  # (N,)
    weights = weights[is_inside]

    deformed = torch.sum(
        weights.unsqueeze(-1) * _to_tensor(cage_verts).unsqueeze(0), dim=1
    )

    is_inside_canonical = (
        cage_voxel.interpolate(points, cage_voxel.voxel_bin, mode="bilinear")[:, 0]
        # > 0.9
        > 0.98
    )  # (N,)
    points[is_inside_canonical] = -1e9

    points[is_inside] = deformed

    return points


def deform_with_MVC(mvc, cage_voxel, points):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        mvc (MeanValue): [num_rays, num_samples along ray, 4]. Optimized mv coordinates.
        cage (Trimesh): [num_rays, num_samples along ray]. Trimesh object.
        points (torch.Tensor): [num_points, 3]. Sampling points.
    Returns:
        deformed: [num_points, 3]. Deformed sampling points.
    """

    cage_verts = cage_voxel.mesh.vertices
    weights = mvc.points_to_weights(points)  # (N,C)

    is_inside = weights.sum(-1) > 0.98  # (N,)

    weights = weights[is_inside]

    deformed = torch.sum(weights.unsqueeze(-1) * _to_tensor(cage_verts).unsqueeze(0), dim=1)

    is_inside_canonical = (
        cage_voxel.interpolate(points, cage_voxel.voxel_bin, mode="bilinear")[:, 0]
        > 0.98
    )  # (N,)
    points[is_inside_canonical] = -1e9

    points[is_inside] = deformed

    return points



def deform_with_GC(gc, cage_voxel, points):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        gc (MeanValue): [num_rays, num_samples along ray, 4]. Optimized green coordinates.
        cage (Trimesh): [num_rays, num_samples along ray]. Trimesh object.
        points (torch.Tensor): [num_points, 3]. Sampling points.
    Returns:
        deformed: [num_points, 3]. Deformed sampling points.
    """

    cage_verts = cage_voxel.mesh.vertices
    weights = gc.points_to_weights(points)  # (N,C)
    add_f = gc.points_to_add_f(points)

    is_inside = weights.sum(-1) > 0.98  # (N,)
    weights = weights[is_inside]
    add_f = add_f[is_inside]

    deformed = torch.sum(weights.unsqueeze(-1) * _to_tensor(cage_verts).unsqueeze(0), dim=1)
    deformed += add_f

    # processing points inside the canonical cage
    is_inside_canonical = (
        cage_voxel.interpolate(points, cage_voxel.voxel_bin, mode="bilinear")[:, 0]
        > 0.98
    )  # (N,)
    points[is_inside_canonical] = -1e9

    points[is_inside] = deformed

    return points