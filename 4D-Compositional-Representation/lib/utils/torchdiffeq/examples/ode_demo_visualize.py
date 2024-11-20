def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu(
            ).numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--',
            t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().numpy().min(), t.cpu().numpy().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()
        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:,
            0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:,
            0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)
        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')
        y, x = np.mgrid[-2:2:21.0j, -2:2:21.0j]
        a = torch.tensor(0).to(device).float()
        b = torch.tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device
            ).float()
        dydt = odefunc(a, b).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = dydt / mag
        dydt = dydt.reshape(21, 21, 2)
        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color=
            'black')
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)
        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)
