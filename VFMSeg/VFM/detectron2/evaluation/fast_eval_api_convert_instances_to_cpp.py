def convert_instances_to_cpp(instances, is_det=False):
    instances_cpp = []
    for instance in instances:
        instance_cpp = _C.InstanceAnnotation(int(instance['id']), instance[
            'score'] if is_det else instance.get('score', 0.0), instance[
            'area'], bool(instance.get('iscrowd', 0)), bool(instance.get(
            'ignore', 0)))
        instances_cpp.append(instance_cpp)
    return instances_cpp
