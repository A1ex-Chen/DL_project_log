def check_voxel_occupied(occupancy_grid):
    occ = occupancy_grid
    occupied = occ[..., :-1, :-1, :-1] & occ[..., :-1, :-1, 1:] & occ[...,
        :-1, 1:, :-1] & occ[..., :-1, 1:, 1:] & occ[..., 1:, :-1, :-1] & occ[
        ..., 1:, :-1, 1:] & occ[..., 1:, 1:, :-1] & occ[..., 1:, 1:, 1:]
    return occupied
