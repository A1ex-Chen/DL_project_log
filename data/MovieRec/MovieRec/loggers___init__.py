def __init__(self, writer, key='train_loss', graph_name='Train Loss',
    group_name='metric'):
    self.key = key
    self.graph_label = graph_name
    self.group_name = group_name
    self.writer = writer
