def make_final_policy(tf_policy_):

    def final_policy(image_):
        for func, prob, args in tf_policy_:
            image_ = _apply_func_with_prob(func, image_, args, prob)
        return image_
    return final_policy
