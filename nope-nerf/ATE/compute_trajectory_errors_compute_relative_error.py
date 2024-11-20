def compute_relative_error(p_es, q_es, p_gt, q_gt, T_cm, dist,
    max_dist_diff, accum_distances=[], scale=1.0):
    if len(accum_distances) == 0:
        accum_distances = tu.get_distance_from_start(p_gt)
    comparisons = tu.compute_comparison_indices_length(accum_distances,
        dist, max_dist_diff)
    n_samples = len(comparisons)
    print('number of samples = {0} '.format(n_samples))
    if n_samples < 2:
        print('Too few samples! Will not compute.')
        return np.array([]), np.array([]), np.array([]), np.array([]
            ), np.array([]), np.array([]), np.array([])
    T_mc = np.linalg.inv(T_cm)
    errors = []
    for idx, c in enumerate(comparisons):
        if not c == -1:
            T_c1 = tu.get_rigid_body_trafo(q_es[idx, :], p_es[idx, :])
            T_c2 = tu.get_rigid_body_trafo(q_es[c, :], p_es[c, :])
            T_c1_c2 = np.dot(np.linalg.inv(T_c1), T_c2)
            T_c1_c2[:3, 3] *= scale
            T_m1 = tu.get_rigid_body_trafo(q_gt[idx, :], p_gt[idx, :])
            T_m2 = tu.get_rigid_body_trafo(q_gt[c, :], p_gt[c, :])
            T_m1_m2 = np.dot(np.linalg.inv(T_m1), T_m2)
            T_m1_m2_in_c1 = np.dot(T_cm, np.dot(T_m1_m2, T_mc))
            T_error_in_c2 = np.dot(np.linalg.inv(T_m1_m2_in_c1), T_c1_c2)
            T_c2_rot = np.eye(4)
            T_c2_rot[0:3, 0:3] = T_c2[0:3, 0:3]
            T_error_in_w = np.dot(T_c2_rot, np.dot(T_error_in_c2, np.linalg
                .inv(T_c2_rot)))
            errors.append(T_error_in_w)
    error_trans_norm = []
    error_trans_perc = []
    error_yaw = []
    error_gravity = []
    e_rot = []
    e_rot_deg_per_m = []
    for e in errors:
        tn = np.linalg.norm(e[0:3, 3])
        error_trans_norm.append(tn)
        error_trans_perc.append(tn / dist * 100)
        ypr_angles = tf.euler_from_matrix(e, 'rzyx')
        e_rot.append(tu.compute_angle(e))
        error_yaw.append(abs(ypr_angles[0]) * 180.0 / np.pi)
        error_gravity.append(np.sqrt(ypr_angles[1] ** 2 + ypr_angles[2] ** 
            2) * 180.0 / np.pi)
        e_rot_deg_per_m.append(e_rot[-1] / dist)
    return errors, np.array(error_trans_norm), np.array(error_trans_perc
        ), np.array(error_yaw), np.array(error_gravity), np.array(e_rot
        ), np.array(e_rot_deg_per_m)
