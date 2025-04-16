def select_and_apply_random_policy(policies: Any, image: tf.Tensor):
    """Select a random policy from `policies` and apply it to `image`."""
    policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf
        .int32)
    for i, policy in enumerate(policies):
        image = tf.cond(tf.equal(i, policy_to_select), lambda
            selected_policy=policy: selected_policy(image), lambda : image)
    return image
