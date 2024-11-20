def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([('exact', 100.0 * sum(exact_scores.
            values()) / total), ('f1', 100.0 * sum(f1_scores.values()) /
            total), ('total', total)])
    else:
        total = len(qid_list)
        return collections.OrderedDict([('exact', 100.0 * sum(exact_scores[
            k] for k in qid_list) / total), ('f1', 100.0 * sum(f1_scores[k] for
            k in qid_list) / total), ('total', total)])
