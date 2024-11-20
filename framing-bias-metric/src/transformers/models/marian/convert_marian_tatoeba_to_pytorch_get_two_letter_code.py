def get_two_letter_code(self, three_letter_code):
    return self.alpha3_to_alpha2.get(three_letter_code, three_letter_code)
