"""
utils.py

"""
import numpy as np

"""
generate temporal probabilities
"""
import string
    




"""
testing temporal predictor on COPA
"""

    premise, choice1, choice2, q, lb = ds[['premise', 'choice1', 'choice2', 'question', 'label']]
    print(f"Premise: {premise}\nC1: {choice1}\nC2: {choice2}\nQuestion: {q}\tCorrect choice: Choice {lb+1}")
    base_res = [base_pred.get_temp(premise, choice1, top_k=top_k), base_pred.get_temp(premise, choice2, top_k=top_k)]
    ft_res = [ft_pred.get_temp(premise, choice1, top_k=top_k), ft_pred.get_temp(premise, choice2, top_k=top_k)]
    exp = " (expect before > after)" if q == 'effect' else " (expect before < after)"
    print(f"\n============== PREMISE <---> CHOICE 1{exp if lb == 0 else ''}\n{premise} <---> {choice1}")
    print(f"Base model:\t{_res_proc(base_res[0])}\nFT model:\t{_res_proc(ft_res[0])}")
    print(f"\n============== PREMISE <---> CHOICE 2{exp if lb == 1 else ''}\n{premise} <---> {choice2}")
    print(f"Base model:\t{_res_proc(base_res[1])}\nFT model:\t{_res_proc(ft_res[1])}")

def test_copa_predict(ds, predictor, top_k=5):
    premise, choice1, choice2, q, lb = ds[['premise', 'choice1', 'choice2', 'question', 'label']]
    res = [predictor.get_temp(premise, choice1, top_k=top_k), predictor.get_temp(premise, choice2, top_k=top_k)]
    befores = [r[0] - r[1] for r in res]
    afters = [r[1] - r[0] for r in res]


    if q == 'effect':
        # wants to find one with higher "AFTER"
        if afters[0] == afters[1]:
            print(f"tie at {premise}/after: {afters[0]}")
            return -1
        return np.argmax(afters)
    else:
        # asks for "cause", wants to find one with higher "BEFORE"
        if befores[0] == befores[1]:
            print(f"tie at {premise}/before: {befores[0]}")
            return -1
        return np.argmax(befores)