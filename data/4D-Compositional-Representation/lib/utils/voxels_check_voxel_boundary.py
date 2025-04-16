def check_voxel_boundary(occupancy_grid):
    occupied = check_voxel_occupied(occupancy_grid)
    unoccupied = check_voxel_unoccupied(occupancy_grid)
    return ~occupied & ~unoccupied
