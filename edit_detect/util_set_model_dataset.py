def set_model_dataset(self, model: str, dataset: str, out_dist: str, real:
    Union[str, List[str]]=None, fake: Union[str, List[str]]=None):
    (self.__model__, self.__real_images__, self.__fake_images__, self.
        __out_dist_real_images__) = (self.prep_model_dataset(model=model,
        dataset=dataset, out_dist=out_dist))
    print(
        f'Model: {self.__model__}, Real: {self.__real_images__}, Fake: {self.__fake_images__}, Out Dist: {self.__out_dist_real_images__}'
        )
    return self
