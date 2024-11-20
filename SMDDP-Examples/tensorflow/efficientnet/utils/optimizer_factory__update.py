def _update(strategy, v_moving_and_v_normal):
    for v_moving, v_normal in v_moving_and_v_normal:
        strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))
