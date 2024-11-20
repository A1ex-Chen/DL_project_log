import torch
import numpy as np
import logging
from matplotlib import pyplot as plt
import os
import shutil
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
logger_py = logging.getLogger(__name__)

























    

# https://github.dev/kwea123/ngp_pl













import scipy.interpolate as si




    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/12, radius)]
    return np.stack(spheric_poses, 0)

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # c = np.dot(c2w[:3,:4], np.array([0.7*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.1, 1.]) * rads)
        # c = np.dot(c2w[:3,:4], np.array([0.3*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.01, 1.]) * rads)
        c = np.dot(c2w[:3,:4], np.array([0.2*np.cos(theta) , -0.2*np.sin(theta) , -np.sin(theta*zrate) *0.1, 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def reprojection(pixels, depth, Rt_ref, world_mat, camera_mat):
   
    assert(pixels.shape[-1] == 2)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    Rt_ref = to_pytorch(Rt_ref)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)
    # Project pixels into camera space
    # pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    pixels_depth = pixels.clone()
    depth = depth.view(1, -1, 1)
    pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    xy_ref = camera_mat @ Rt_ref @ torch.inverse(world_mat) @ torch.inverse(camera_mat) @ pixels_depth

    # Transform p_world back to 3D coordinates
    xy_ref = xy_ref[:, :3].permute(0, 2, 1)
    xy_ref = xy_ref[..., :2] / xy_ref[..., 2:]
    
    valid_points = xy_ref.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(-1).float()
    if is_numpy:
        xy_ref = xy_ref.numpy()
    return xy_ref, valid_mask
def project_to_cam(points, camera_mat, device):
    '''
    points: (B, N, 3)
    camera_mat: (B, 4, 4)
    '''
    # breakpoint()
    B, N, D = points.size()
    points, is_numpy = to_pytorch(points, True)
    points = points.permute(0, 2, 1)
    points = torch.cat([points, torch.ones(B, 1, N, device=device)], dim=1)

    xy_ref = camera_mat @ points

    xy_ref = xy_ref[:, :3].permute(0, 2, 1)
    xy_ref = xy_ref[..., :2] / xy_ref[..., 2:]
    
    
    valid_points = xy_ref.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(-1).bool()
    if is_numpy:
        xy_ref = xy_ref.numpy()
    return xy_ref, valid_mask

def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                        torch.stack([w2,O,-w0],dim=-1),
                        torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx

def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def backup(out_dir, config):
    backup_path = os.path.join(out_dir, 'backup')
    if not os.path.exists(backup_path):
            os.makedirs(backup_path)
    shutil.copyfile(config, os.path.join(backup_path, 'config.yaml'))
    shutil.copy('train.py', backup_path)
    shutil.copy('./configs/default.yaml', backup_path)
    base_dirs = ['./model', './dataloading']
    for base_dir in base_dirs:
        files_ = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir,f))]
        backup_subpath = os.path.join(backup_path, base_dir[2:])
        if not os.path.exists(backup_subpath):
            os.makedirs(backup_subpath)
        for file in files_:
            shutil.copy(os.path.join(base_dir, file), backup_subpath)




def interp_poses(c2ws, N_views):
    N_inputs = c2ws.shape[0]
    trans = c2ws[:, :3, 3:].permute(2, 1, 0)
    rots = c2ws[:, :3, :3]
    render_poses = []
    rots = R.from_matrix(rots)
    slerp = Slerp(np.linspace(0, 1, N_inputs), rots)
    interp_rots = torch.tensor(slerp(np.linspace(0, 1, N_views)).as_matrix().astype(np.float32))
    interp_trans = torch.nn.functional.interpolate(trans, size=N_views, mode='linear').permute(2, 1, 0)
    render_poses = torch.cat([interp_rots, interp_trans], dim=2)
    render_poses = convert3x4_4x4(render_poses)
    return render_poses
def interp_poses_bspline(c2ws, N_novel_imgs, input_times, degree):
    target_trans = torch.tensor(scipy_bspline(c2ws[:, :3, 3],n=N_novel_imgs,degree=degree,periodic=False).astype(np.float32)).unsqueeze(2)
    rots = R.from_matrix(c2ws[:, :3, :3])
    slerp = Slerp(input_times, rots)
    target_times = np.linspace(input_times[0], input_times[-1], N_novel_imgs)
    target_rots = torch.tensor(slerp(target_times).as_matrix().astype(np.float32))
    target_poses = torch.cat([target_rots, target_trans], dim=2)
    target_poses = convert3x4_4x4(target_poses)
    return target_poses

def get_poses_at_times(c2ws, input_times, target_times):
    trans = c2ws[:, :3, 3:]
    rots = c2ws[:, :3, :3]
    N_target = len(target_times)
    rots = R.from_matrix(rots)
    slerp = Slerp(input_times, rots)
    target_rots = torch.tensor(slerp(target_times).as_matrix().astype(np.float32))
    target_trans = interp_t(trans, input_times, target_times)
    target_poses = torch.cat([target_rots, target_trans], dim=2)
    target_poses = convert3x4_4x4(target_poses)
    return target_poses
def interp_t(trans, input_times, target_times):
    target_trans = []
    for target_t in target_times:
        diff = target_t - input_times
        array1 = diff.copy()
        array1[diff<0] = 1000
        array2 = diff.copy()
        array2[diff>0] = -1000
        t1_idx = np.argmin(array1)
        t2_idx = np.argmin(-array2)
        target_tran = (target_t-input_times[t1_idx])/(input_times[t2_idx] - input_times[t1_idx]) * trans[t1_idx] + \
                (input_times[t2_idx]- target_t)/(input_times[t2_idx] - input_times[t1_idx])* trans[t2_idx]
        target_trans.append(target_tran)
    target_trans = torch.stack(target_trans, axis=0)
    return target_trans


import scipy.interpolate as si

def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree,count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
        degree = np.clip(degree,1,degree)

    # Opened curve
    else:
        degree = np.clip(degree,1,count-1)
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0,max_param,n))

def generate_spiral_nerf(learned_poses, bds, N_novel_views, hwf):
    learned_poses_ = np.concatenate((learned_poses[:,:3,:4].detach().cpu().numpy(), hwf[:len(learned_poses)]), axis=-1)
    c2w = poses_avg(learned_poses_)
    print('recentered', c2w.shape)
    # Get spiral
    # Get average pose
    up = normalize(learned_poses_[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset
    
    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = learned_poses_[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_rots = 2
    c2ws = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_novel_views)
    c2ws = torch.tensor(np.stack(c2ws).astype(np.float32))
    c2ws = c2ws[:,:3,:4]
    return c2ws
def convert2mip(pts):
    # pts
    pts_norm = torch.linalg.norm(pts, ord=2, dim=-1)
    outside_mask = (pts_norm >= 1.0)
    mip_pts = pts.clone()
    mip_pts[outside_mask,:] = (2 - 1.0 / pts_norm[outside_mask, None]) * (pts[outside_mask,:] / pts_norm[outside_mask, None])
    return mip_pts
def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

def get_ndc_rays_fxfy(fxfy, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (1 / fxfy[0]) * ox_oz
    o1 = -1. / (1/ fxfy[1]) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (1 / fxfy[0]) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (1 / fxfy[1]) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3