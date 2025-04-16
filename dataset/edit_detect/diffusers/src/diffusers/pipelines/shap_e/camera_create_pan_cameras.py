def create_pan_cameras(size: int) ->DifferentiableProjectiveCamera:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z ** 2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableProjectiveCamera(origin=torch.from_numpy(np.stack(
        origins, axis=0)).float(), x=torch.from_numpy(np.stack(xs, axis=0))
        .float(), y=torch.from_numpy(np.stack(ys, axis=0)).float(), z=torch
        .from_numpy(np.stack(zs, axis=0)).float(), width=size, height=size,
        x_fov=0.7, y_fov=0.7, shape=(1, len(xs)))
