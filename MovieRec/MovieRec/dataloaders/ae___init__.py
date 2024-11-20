def __init__(self, user2items, item_count):
    input_list, label_list = self.split_input_label_proportion(user2items)
    input_user_row, label_user_row = [], []
    for user, input_items in enumerate(input_list):
        for _ in range(len(input_items)):
            input_user_row.append(user)
    for user, label_items in enumerate(label_list):
        for _ in range(len(label_items)):
            label_user_row.append(user)
    input_user_row, label_user_row = np.array(input_user_row), np.array(
        label_user_row)
    input_item_col = np.hstack(input_list)
    label_item_col = np.hstack(label_list)
    sparse_input = sparse.csr_matrix((np.ones(len(input_user_row)), (
        input_user_row, input_item_col)), dtype='float64', shape=(len(
        input_list), item_count))
    sparse_label = sparse.csr_matrix((np.ones(len(label_user_row)), (
        label_user_row, label_item_col)), dtype='float64', shape=(len(
        label_list), item_count))
    self.input_data = torch.FloatTensor(sparse_input.toarray())
    self.label_data = torch.FloatTensor(sparse_label.toarray())
