def ablation_query(dt, D, F, S, Q, C, E):
    return dt.query(
        f'(normalization == {S}) and (direct_match=={D}) and (use_cooccur=={C}) and (temp_filter=={F}) and (res_norm=={E}) and (pxa_norm=={Q})'
        )
