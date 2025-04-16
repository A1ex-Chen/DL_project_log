def update_sequence(self, request):
    if request.sequence_number <= self.sequence_number:
        return
    self.sequence_number = request.sequence_number
