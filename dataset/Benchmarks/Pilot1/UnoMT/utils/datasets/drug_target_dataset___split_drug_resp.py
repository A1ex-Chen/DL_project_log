def __split_drug_resp(self):
    training_df, validation_df = train_test_split(self.__drug_target_df,
        test_size=self.__validation_ratio, random_state=self.__rand_state,
        shuffle=True)
    self.__drug_target_df = training_df if self.training else validation_df
