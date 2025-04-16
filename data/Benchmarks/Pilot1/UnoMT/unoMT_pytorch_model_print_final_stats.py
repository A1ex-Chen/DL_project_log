def print_final_stats(self):
    args = self.args
    val_cl_clf_acc = np.array(self.val_cl_clf_acc).reshape(-1, 3)
    val_resp_mse, val_resp_mae, val_resp_r2 = np.array(self.val_resp_mse
        ).reshape(-1, len(args.val_sources)), np.array(self.val_resp_mae
        ).reshape(-1, len(args.val_sources)), np.array(self.val_resp_r2
        ).reshape(-1, len(args.val_sources))
    print('Program Running Time: %.1f Seconds.' % (time.time() - self.
        start_time))
    print('=' * 80)
    print('Overall Validation Results:\n')
    print('\tBest Results from Different Models (Epochs):')
    clf_targets = ['Cell Line Categories', 'Cell Line Sites', 'Cell Line Types'
        ]
    best_acc = np.amax(val_cl_clf_acc, axis=0)
    best_acc_epochs = np.argmax(val_cl_clf_acc, axis=0)
    for index, clf_target in enumerate(clf_targets):
        print('\t\t%-24s Best Accuracy: %.3f%% (Epoch = %3d)' % (clf_target,
            best_acc[index], best_acc_epochs[index] + 1 + args.
            resp_val_start_epoch))
    print('\t\tDrug Target Family \t Best Accuracy: %.3f%% (Epoch = %3d)' %
        (np.max(self.val_drug_target_acc), np.argmax(self.
        val_drug_target_acc) + 1 + args.resp_val_start_epoch))
    print(
        '\t\tDrug Weighted QED \t Best R2 Score: %+6.4f (Epoch = %3d, MSE = %8.6f, MAE = %8.6f)'
         % (np.max(self.val_drug_qed_r2), np.argmax(self.val_drug_qed_r2) +
        1 + args.resp_val_start_epoch, self.val_drug_qed_mse[np.argmax(self
        .val_drug_qed_r2)], self.val_drug_qed_mae[np.argmax(self.
        val_drug_qed_r2)]))
    val_data_sources = [loader.dataset.data_source for loader in self.
        drug_resp_val_loaders]
    best_r2 = np.amax(self.val_resp_r2, axis=0)
    best_r2_epochs = np.argmax(self.val_resp_r2, axis=0)
    for index, data_source in enumerate(val_data_sources):
        print(
            '\t\t%-6s \t Best R2 Score: %+6.4f (Epoch = %3d, MSE = %8.2f, MAE = %6.2f)'
             % (data_source, best_r2[index], best_r2_epochs[index] + args.
            resp_val_start_epoch + 1, val_resp_mse[best_r2_epochs[index],
            index], val_resp_mae[best_r2_epochs[index], index]))
    best_epoch = val_resp_r2[:, self.val_index].argmax()
    print('\n\tBest Results from the Same Model (Epoch = %3d):' % (
        best_epoch + 1 + args.resp_val_start_epoch))
    for index, clf_target in enumerate(clf_targets):
        print('\t\t%-24s Accuracy: %.3f%%' % (clf_target, val_cl_clf_acc[
            best_epoch, index]))
    print('\t\tDrug Target Family \t Accuracy: %.3f%% ' % self.
        val_drug_target_acc[best_epoch])
    print(
        '\t\tDrug Weighted QED \t R2 Score: %+6.4f (MSE = %8.6f, MAE = %6.6f)'
         % (self.val_drug_qed_r2[best_epoch], self.val_drug_qed_mse[
        best_epoch], self.val_drug_qed_mae[best_epoch]))
    for index, data_source in enumerate(val_data_sources):
        print('\t\t%-6s \t R2 Score: %+6.4f (MSE = %8.2f, MAE = %6.2f)' % (
            data_source, val_resp_r2[best_epoch, index], val_resp_mse[
            best_epoch, index], val_resp_mae[best_epoch, index]))
