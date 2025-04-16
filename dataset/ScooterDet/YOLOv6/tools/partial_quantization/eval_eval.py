def eval(self, model):
    model.eval()
    model.to(self.device)
    if self.half is True:
        model.half()
    with torch.no_grad():
        pred_result, vis_outputs, vis_paths = self.val.predict_model(model,
            self.val_loader, self.task)
        eval_result = self.val.eval_model(pred_result, model, self.
            val_loader, self.task)
    return eval_result
