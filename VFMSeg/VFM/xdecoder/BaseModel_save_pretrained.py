def save_pretrained(self, save_dir):
    save_path = os.path.join(save_dir, 'model_state_dict.pt')
    torch.save(self.model.state_dict(), save_path)
