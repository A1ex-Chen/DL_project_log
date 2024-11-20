def custom_combo_score(combined_growth, growth_1, growth_2):
    if growth_1 <= 0 or growth_2 <= 0:
        expected_growth = min(growth_1, growth_2)
    else:
        expected_growth = growth_1 * growth_2
    custom_score = (expected_growth - combined_growth) * 100
    return custom_score
