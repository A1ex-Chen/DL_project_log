def update_metric(best_metric, new_metric):
    for key in new_metric:
        if key not in best_metric:
            best_metric[key] = new_metric[key]
        else:
            best_metric[key] = max(best_metric[key], new_metric[key])
    return best_metric
