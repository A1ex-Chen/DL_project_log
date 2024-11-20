def check_voxel_unoccupied(occupancy_grid):
    occ = occupancy_grid
    unoccupied = ~(occ[..., :-1, :-1, :-1] | occ[..., :-1, :-1, 1:] | occ[
        ..., :-1, 1:, :-1] | occ[..., :-1, 1:, 1:] | occ[..., 1:, :-1, :-1] |
        occ[..., 1:, :-1, 1:] | occ[..., 1:, 1:, :-1] | occ[..., 1:, 1:, 1:])
    return unoccupied
