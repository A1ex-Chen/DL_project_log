def generate_spiral2d(nspiral=1000, ntotal=500, nsample=100, start=0.0,
    stop=1, noise_std=0.1, a=0.0, b=1.0, savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]
    zs_cw = stop + 1.0 - orig_ts
    rs_cw = a + b * 50.0 / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5.0, rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)
    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5.0, rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)
    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        t0_idx = npr.multinomial(1, [1.0 / (ntotal - 2.0 * nsample)] * (
            ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample
        cc = bool(npr.rand() > 0.5)
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)
        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    return orig_trajs, samp_trajs, orig_ts, samp_ts
