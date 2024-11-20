def test_copa_run(ds, base_pred, ft_pred, top_k=5):

    def _res_proc(res):
        return f'before: {res[0]:.3f}\tafter: {res[1]:.3f}'
    premise, choice1, choice2, q, lb = ds[['premise', 'choice1', 'choice2',
        'question', 'label']]
    print(
        f"""Premise: {premise}
C1: {choice1}
C2: {choice2}
Question: {q}	Correct choice: Choice {lb + 1}"""
        )
    base_res = [base_pred.get_temp(premise, choice1, top_k=top_k),
        base_pred.get_temp(premise, choice2, top_k=top_k)]
    ft_res = [ft_pred.get_temp(premise, choice1, top_k=top_k), ft_pred.
        get_temp(premise, choice2, top_k=top_k)]
    exp = (' (expect before > after)' if q == 'effect' else
        ' (expect before < after)')
    print(
        f"""
============== PREMISE <---> CHOICE 1{exp if lb == 0 else ''}
{premise} <---> {choice1}"""
        )
    print(
        f'Base model:\t{_res_proc(base_res[0])}\nFT model:\t{_res_proc(ft_res[0])}'
        )
    print(
        f"""
============== PREMISE <---> CHOICE 2{exp if lb == 1 else ''}
{premise} <---> {choice2}"""
        )
    print(
        f'Base model:\t{_res_proc(base_res[1])}\nFT model:\t{_res_proc(ft_res[1])}'
        )
