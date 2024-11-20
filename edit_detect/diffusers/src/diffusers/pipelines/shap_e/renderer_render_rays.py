@torch.no_grad()
def render_rays(self, rays, sampler, n_samples, prev_model_out=None,
    render_with_direction=False):
    """
        Perform volumetric rendering over a partition of possible t's in the union of rendering volumes (written below
        with some abuse of notations)

            C(r) := sum(
                transmittance(t[i]) * integrate(
                    lambda t: density(t) * channels(t) * transmittance(t), [t[i], t[i + 1]],
                ) for i in range(len(parts))
            ) + transmittance(t[-1]) * void_model(t[-1]).channels

        where

        1) transmittance(s) := exp(-integrate(density, [t[0], s])) calculates the probability of light passing through
        the volume specified by [t[0], s]. (transmittance of 1 means light can pass freely) 2) density and channels are
        obtained by evaluating the appropriate part.model at time t. 3) [t[i], t[i + 1]] is defined as the range of t
        where the ray intersects (parts[i].volume \\ union(part.volume for part in parts[:i])) at the surface of the
        shell (if bounded). If the ray does not intersect, the integral over this segment is evaluated as 0 and
        transmittance(t[i + 1]) := transmittance(t[i]). 4) The last term is integration to infinity (e.g. [t[-1],
        math.inf]) that is evaluated by the void_model (i.e. we consider this space to be empty).

        args:
            rays: [batch_size x ... x 2 x 3] origin and direction. sampler: disjoint volume integrals. n_samples:
            number of ts to sample. prev_model_outputs: model outputs from the previous rendering step, including

        :return: A tuple of
            - `channels`
            - A importance samplers for additional fine-grained rendering
            - raw model output
        """
    origin, direction = rays[..., 0, :], rays[..., 1, :]
    vrange = self.volume.intersect(origin, direction, t0_lower=None)
    ts = sampler.sample(vrange.t0, vrange.t1, n_samples)
    ts = ts.to(rays.dtype)
    if prev_model_out is not None:
        ts = torch.sort(torch.cat([ts, prev_model_out.ts], dim=-2), dim=-2
            ).values
    batch_size, *_shape, _t0_dim = vrange.t0.shape
    _, *ts_shape, _ts_dim = ts.shape
    directions = torch.broadcast_to(direction.unsqueeze(-2), [batch_size, *
        ts_shape, 3])
    positions = origin.unsqueeze(-2) + ts * directions
    directions = directions.to(self.mlp.dtype)
    positions = positions.to(self.mlp.dtype)
    optional_directions = directions if render_with_direction else None
    model_out = self.mlp(position=positions, direction=optional_directions,
        ts=ts, nerf_level='coarse' if prev_model_out is None else 'fine')
    channels, weights, transmittance = integrate_samples(vrange, model_out.
        ts, model_out.density, model_out.channels)
    transmittance = torch.where(vrange.intersected, transmittance, torch.
        ones_like(transmittance))
    channels = torch.where(vrange.intersected, channels, torch.zeros_like(
        channels))
    channels = channels + transmittance * self.void(origin)
    weighted_sampler = ImportanceRaySampler(vrange, ts=model_out.ts,
        weights=weights)
    return channels, weighted_sampler, model_out
