def forward(self, input, mixup_lambda=None):
    """Input: (batch_size, data_length)"""
    output_dict = self.base(input, mixup_lambda)
    embedding = output_dict['embedding']
    clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
    output_dict['clipwise_output'] = clipwise_output
    return output_dict
