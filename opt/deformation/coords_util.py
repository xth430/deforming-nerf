# Borrowed from https://github.com/yifita/deep_cage and https://github.com/yifita/pytorch_points
import math
import torch
import numpy as np


class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, out_size, fill=0.0):
        out = torch.full(out_size, fill, device=src.device, dtype=src.dtype)
        ctx.save_for_backward(idx)
        out.scatter_add_(dim, idx, src)
        ctx.mark_non_differentiable(idx)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, ograd):
        idx, = ctx.saved_tensors
        grad = torch.gather(ograd, ctx.dim, idx)
        return grad, None, None, None, None

_scatter_add = ScatterAdd.apply

def scatter_add(src, idx, dim, out_size=None, fill=0.0):
    if out_size is None:
        out_size = list(src.size())
        dim_size = idx.max().item()+1
        out_size[dim] = dim_size
    return _scatter_add(src, idx, dim, out_size, fill)


def normalize(tensor, dim=-1):
    """normalize tensor in specified dimension"""
    return torch.nn.functional.normalize(tensor, p=2, dim=dim, eps=1e-12, out=None)


def check_values(tensor):
    """return true if tensor doesn't contain NaN or Inf"""
    return not (torch.any(torch.isnan(tensor)).item() or torch.any(torch.isinf(tensor)).item())


def mean_value_coordinates_3D(query, vertices, faces, verbose=False, check_values_bool=True):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        query    (B,P,3)
        vertices (B,N,3)
        faces    (B,F,3)
    return:
        wj       (B,P,N)
    """
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices.unsqueeze(1) - query.unsqueeze(2)
    # \|u_i\| (B,P,N,1)
    dj = torch.norm(uj, dim=-1, p=2, keepdim=True)
    uj = normalize(uj, dim=-1)
    # gather triangle B,P,F,3,3
    ui = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
                                   3,
                                   faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = torch.norm(ui[:,:,:,[1, 2, 0],:] - ui[:, :, :,[2, 0, 1],:], dim=-1, p=2)
    eps = 2e-5
    li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*torch.asin(li/2)
    if check_values_bool:
        assert(check_values(theta_i))

    del li

    # B,P,F,1
    h = torch.sum(theta_i, dim=-1, keepdim=True)/2

    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]))-1
    # ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]) + 1e-15)-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)

    si = torch.sign(torch.det(ui)).unsqueeze(-1)*torch.sqrt(1-ci**2)  # sqrt gradient nan for 0
    if check_values_bool:
        assert(check_values(si))
    # (B,P,F,3)
    di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
                      faces.unsqueeze(1).expand(-1,P,-1,-1))
    if check_values_bool:
        assert(check_values(di))
    # if si.requires_grad:
    #     vertices.register_hook(save_grad("mvc/dv"))
    #     li.register_hook(save_grad("mvc/dli"))
    #     theta_i.register_hook(save_grad("mvc/dtheta"))
    #     ci.register_hook(save_grad("mvc/dci"))
    #     si.register_hook(save_grad("mvc/dsi"))
    #     di.register_hook(save_grad("mvc/ddi"))

    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    # CHECK is there a 2* in the denominator
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])
    # wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]] + 1e-15)
    # if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # ignore coplaner outside triangle
    # alternative check
    # (B,F,3,3)
    # triangle_points = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))
    # # (B,P,F,3), (B,1,F,3) -> (B,P,F,1)
    # determinant = dot_product(triangle_points[:,:,:,0].unsqueeze(1)-query.unsqueeze(2),
    #                           torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0],
    #                                       triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1).unsqueeze(1), dim=-1, keepdim=True).detach()
    # # (B,P,F,1)
    # sqrdist = determinant*determinant / (4 * sqrNorm(torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0], triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1), keepdim=True))

    wi = torch.where(torch.any(torch.abs(si) <= 1e-5, keepdim=True, dim=-1), torch.zeros_like(wi), wi)
    # wi = torch.where(sqrdist <= 1e-5, torch.zeros_like(wi), wi)

    # if π −h < ε, x lies on t, use 2D barycentric coordinates
    # inside triangle
    inside_triangle = (math.pi-h).squeeze(-1)<1e-4
    # set all F for this P to zero
    wi = torch.where(torch.any(inside_triangle, dim=-1, keepdim=True).unsqueeze(-1), torch.zeros_like(wi), wi)
    # CHECK is it di https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf or li http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf
    wi = torch.where(inside_triangle.unsqueeze(-1).expand(-1,-1,-1,wi.shape[-1]), torch.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    if check_values_bool:
        assert(check_values(wi))

    # sum over all faces face -> vertex (B,P,F*3) -> (B,P,N)
    wj = scatter_add(wi.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))

    if check_values_bool:
        assert(check_values(wj))


    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = torch.where(torch.any(close_to_point, dim=-1, keepdim=True), torch.zeros_like(wj), wj)
    wj = torch.where(close_to_point, torch.ones_like(wj), wj)

    # (B,P,1)
    sumWj = torch.sum(wj, dim=-1, keepdim=True)
    sumWj = torch.where(sumWj==0, torch.ones_like(sumWj), sumWj)

    wj_normalised = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["mvc/wi"] = wi
    #     wi.register_hook(save_grad("mvc/dwi"))
    #     wj.register_hook(save_grad("mvc/dwj"))
    if verbose:
        return wj_normalised, wi
    else:
        return wj_normalised



def green_coordinates_3D(query, vertices, faces, face_normals=None, verbose=False):
    """
    Lipman et.al. sum_{i\in N}(phi_i*v_i)+sum_{j\in F}(psi_j*n_j)
    http://www.wisdom.weizmann.ac.il/~ylipman/GC/CageMesh_GreenCoords.cpp
    params:
        query    (B,P,D), D=3
        vertices (B,N,D), D=3
        faces    (B,F,3)
    return:
        phi_i    (B,P,N)
        psi_j    (B,P,F)
        exterior_flag (B,P)
    """
    B, F, _ = faces.shape
    _, P, D = query.shape
    _, N, D = vertices.shape
    # (B,F,D)
    n_t = face_normals
    if n_t is None:
        # compute face normal
        n_t, _ = compute_face_normals_and_areas(vertices, faces)

    vertices = vertices.detach()
    # (B,N,D) (B,F,3) -> (B,F,3,3) face points
    v_jl = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))

    # v_jl = v_jl - x (B,P,F,3,3)
    v_jl = v_jl.view(B,1,F,3,3) - query.view(B,P,1,1,3)
    # (B,P,F,D).(B,1,F,D) -> (B,P,F,1)*(B,P,F,D) projection of v1_x on the normal
    p = dot_product(v_jl[:,:,:,0,:], n_t.unsqueeze(1).expand(-1,P,-1,-1), dim=-1, keepdim=True)*n_t.unsqueeze(1)

    # B,P,F,3,D -> B,P,F,3
    s_l = torch.sign(dot_product(torch.cross(v_jl-p.unsqueeze(-2), v_jl[:,:,:,[1,2,0],:]-p.unsqueeze(-2), dim=-1), n_t.view(B,1,F,1,D)))
    # import pdb; pdb.set_trace()
    # (B,P,F,3)
    I_l = _gcTriInt(p, v_jl, v_jl[:,:,:,[1,2,0],:], None)
    # (B,P,F)
    I = -torch.abs(torch.sum(s_l*I_l, dim=-1))
    GC_face = -I
    assert(check_values(GC_face))
    II_l = _gcTriInt(torch.zeros_like(p), v_jl[:,:,:,[1,2,0], :], v_jl, None)
    # (B,P,F,3,D)
    N_l = torch.cross(v_jl[:,:,:,[1,2,0],:], v_jl, dim=-1)
    N_l_norm = torch.norm(N_l, dim=-1, p=2)
    II_l.masked_fill_(N_l_norm<1e-7, 0)
    # normalize but ignore those with small norms
    N_l = torch.where((N_l_norm>1e-7).unsqueeze(-1), N_l/N_l_norm.unsqueeze(-1), N_l)
    # (B,P,F,D)
    omega = n_t.unsqueeze(1)*I.unsqueeze(-1)+torch.sum(N_l*II_l.unsqueeze(-1), dim=-2)
    eps = 1e-6
    # (B,P,F,3)
    phi_jl = dot_product(N_l[:,:,:,[1,2,0],:], omega.unsqueeze(-2), dim=-1)/(dot_product(N_l[:,:,:,[1,2,0],:], v_jl, dim=-1)+1e-10)
    # on the same plane don't contribute to phi
    phi_jl.masked_fill_((torch.norm(omega, p=2, dim=-1)<eps).unsqueeze(-1), 0)
    # sum per face weights to per vertex weights
    GC_vertex = scatter_add(phi_jl.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))
    assert(check_values(GC_vertex))

    # NOTE the point is inside the face, remember factor 2
    # insideFace = (torch.norm(omega,dim=-1)<1e-5)&torch.all(s_l>0,dim=-1)
    # phi_jl = torch.where(insideFace.unsqueeze(-1), phi_jl, torch.zeros_like(phi_jl))

    # normalize
    sumGC_V = torch.sum(GC_vertex, dim=2, keepdim=True)

    exterior_flag = sumGC_V<0.5

    GC_vertex = GC_vertex/(sumGC_V+1e-10)
    # GC_vertex.masked_fill_(sumGC_V.abs()<eps, 0.0)

    return GC_vertex, GC_face, exterior_flag


def _gcTriInt(p, v1, v2, x):
    """
    part of the gree coordinate 3D pseudo code
    params:
        p  (B,P,F,3)
        v1 (B,P,F,3,3)
        v2 (B,P,F,3,3)
        x  (B,P,F,3)
    return:
        (B,P,F,3)
    """
    eps = 1e-6
    angle_eps = 1e-3
    div_guard = 1e-12
    # (B,P,F,3,D)
    p_v1 = p.unsqueeze(-2)-v1
    v2_p = v2-p.unsqueeze(-2)
    v2_v1 = v2-v1
    # (B,P,F,3)
    p_v1_norm = torch.norm(p_v1, dim=-1, p=2)
    # (B,P,F,3)
    tempval = dot_product(v2_v1, p_v1, dim=-1)/(p_v1_norm*torch.norm(v2_v1, dim=-1, p=2)+div_guard)
    tempval.clamp_(-1.0,1.0)
    filter_mask = tempval.abs()>(1-eps)
    tempval.clamp_(-1.0+eps,1.0-eps)
    alpha = torch.acos(tempval)
    filter_mask = filter_mask | (torch.abs(alpha-np.pi)<angle_eps)|(torch.abs(alpha)<angle_eps)

    tempval = dot_product(-p_v1, v2_p, dim=-1)/(p_v1_norm*torch.norm(v2_p, dim=-1, p=2)+div_guard)
    tempval.clamp_(-1.0, 1.0)
    filter_mask = filter_mask|(torch.abs(tempval)>(1-eps))
    tempval.clamp_(-1.0+eps,1.0-eps)
    beta = torch.acos(tempval)
    assert(check_values(alpha))
    assert(check_values(beta))
    # (B,P,F,3)
    lambd = (p_v1_norm*torch.sin(alpha))**2
    # c (B,P,F,1)
    if x is not None:
        c = torch.sum((p-x)*(p-x), dim=-1,keepdim=True)
    else:
        c = torch.sum(p*p, dim=-1,keepdim=True)
    # theta in (pi-alpha, pi-alpha-beta)
    # (B,P,F,3)
    theta_1 = torch.clamp(np.pi - alpha, 0, np.pi)
    theta_2 = torch.clamp(np.pi - alpha - beta, -np.pi, np.pi)

    S_1, S_2 = torch.sin(theta_1), torch.sin(theta_2)
    C_1, C_2 = torch.cos(theta_1), torch.cos(theta_2)
    sqrt_c = torch.sqrt(c+div_guard)
    sqrt_lmbd = torch.sqrt(lambd+div_guard)
    theta_half = theta_1/2
    filter_mask = filter_mask | ((C_1-1).abs()<eps)
    sqcot_1 = torch.where((C_1-1).abs()<eps, torch.zeros_like(C_1), S_1*S_1/((1-C_1)**2+div_guard))
    # sqcot_1 = torch.where(theta_half.abs()<angle_eps, torch.zeros_like(theta_half), 1/(torch.tan(theta_half)**2+div_guard))
    theta_half = theta_2/2
    filter_mask = filter_mask | ((C_2-1).abs()<eps)
    sqcot_2 = torch.where((C_2-1).abs()<eps, torch.zeros_like(C_2), S_2*S_2/((1-C_2)**2+div_guard))
    # sqcot_2 = torch.where(theta_half.abs()<angle_eps, torch.zeros_like(theta_half), 1/(torch.tan(theta_half)**2+div_guard))
    # I=-0.5*Sign(sx)* ( 2*sqrtc*atan((sqrtc*cx) / (sqrt(a+c*sx*sx) ) )+
    #                 sqrta*log(((sqrta*(1-2*c*cx/(c*(1+cx)+a+sqrta*sqrt(a+c*sx*sx)))))*(2*sx*sx/pow((1-cx),2))))
    # assign a value to invalid entries, backward
    inLog = sqrt_lmbd*(1-2*c*C_1/( div_guard +c*(1+C_1)+lambd+sqrt_lmbd*torch.sqrt(lambd+c*S_1*S_1+div_guard) ) )*2*sqcot_1
    inLog.masked_fill_(filter_mask | (inLog<=0), 1.0)
    # inLog = torch.where(invalid_values|(lambd==0), torch.ones_like(theta_1), div_guard +sqrt_lmbd*(1-2*c*C_1/( div_guard +c*(1+C_1)+lambd+sqrt_lmbd*torch.sqrt(lambd+c*S_1*S_1)+div_guard ) )*2*cot_1)
    I_1 = -0.5*torch.sign(S_1)*(2*sqrt_c*torch.atan((sqrt_c*C_1) / (torch.sqrt(lambd+S_1*S_1*c+div_guard) ) )+sqrt_lmbd*torch.log(inLog))
    assert(check_values(I_1))
    inLog = sqrt_lmbd*(1-2*c*C_2/( div_guard +c*(1+C_2)+lambd+sqrt_lmbd*torch.sqrt(lambd+c*S_2*S_2+div_guard) ) )*2*sqcot_2
    inLog.masked_fill_(filter_mask | (inLog<=0), 1.0)
    I_2 = -0.5*torch.sign(S_2)*(2*sqrt_c*torch.atan((sqrt_c*C_2) / (torch.sqrt(lambd+S_2*S_2*c+div_guard) ) )+sqrt_lmbd*torch.log(inLog))
    assert(check_values(I_2))
    myInt = -1/(4*np.pi)*torch.abs(I_1-I_2-sqrt_c*beta)
    myInt.masked_fill_(filter_mask, 0.0)
    return myInt

def dot_product(tensor1, tensor2, dim=-1, keepdim=False):
    return torch.sum(tensor1*tensor2, dim=dim, keepdim=keepdim)

def compute_face_normals_and_areas(vertices: torch.Tensor, faces: torch.Tensor):
    """
    :params
        vertices   (B,N,3)
        faces      (B,F,3)
    :return
        face_normals         (B,F,3)
        face_areas   (B,F)
    """
    ndim = vertices.ndimension()
    if vertices.ndimension() == 2 and faces.ndimension() == 2:
        vertices.unsqueeze_(0)
        faces.unsqueeze_(0)

    B,N,D = vertices.shape
    F = faces.shape[1]
    # (B,F*3,3)
    face_vertices = torch.gather(vertices, 1, faces.view(B, -1, 1).expand(-1, -1, D)).view(B,F,3,D)
    face_normals = torch.cross(face_vertices[:,:,1,:] - face_vertices[:,:,0,:],
                               face_vertices[:,:,2,:] - face_vertices[:,:,1,:], dim=-1)
    face_areas = face_normals.clone()
    face_areas = torch.sqrt((face_areas ** 2).sum(dim=-1))
    face_areas /= 2
    face_normals = normalize(face_normals, dim=-1)
    if ndim == 2:
        vertices.squeeze_(0)
        faces.squeeze_(0)
        face_normals.squeeze_(0)
        face_areas.squeeze_(0)
    # assert (not np.any(face_areas.unsqueeze(-1) == 0)), 'has zero area face: %s' % mesh.filename
    return face_normals, face_areas