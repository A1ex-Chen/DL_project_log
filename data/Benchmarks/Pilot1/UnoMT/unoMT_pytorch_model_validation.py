def validation(self, epoch):
    device = self.device
    cl_category_acc, cl_site_acc, cl_type_acc = valid_cl_clf(device=device,
        category_clf_net=self.category_clf_net, site_clf_net=self.
        site_clf_net, type_clf_net=self.type_clf_net, data_loader=self.
        cl_clf_val_loader)
    self.val_cl_clf_acc.append([cl_category_acc, cl_site_acc, cl_type_acc])
    drug_target_acc = valid_drug_target(device=device, drug_target_net=self
        .drug_target_net, data_loader=self.drug_target_val_loader)
    self.val_drug_target_acc.append(drug_target_acc)
    drug_qed_mse, drug_qed_mae, drug_qed_r2 = valid_drug_qed(device=device,
        drug_qed_net=self.drug_qed_net, data_loader=self.drug_qed_val_loader)
    self.val_drug_qed_mse.append(drug_qed_mse)
    self.val_drug_qed_mae.append(drug_qed_mae)
    self.val_drug_qed_r2.append(drug_qed_r2)
    resp_mse, resp_mae, resp_r2 = valid_resp(device=device, resp_net=self.
        resp_net, data_loaders=self.drug_resp_val_loaders)
    self.val_resp_mse.append(resp_mse)
    self.val_resp_mae.append(resp_mae)
    self.val_resp_r2.append(resp_r2)
    return resp_r2
