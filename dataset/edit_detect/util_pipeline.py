@property
def pipeline(self):
    if self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_DDPM,
        ModelDataset.MD_DDPM_EMA]):
        return DDPMPipeline.from_pretrained(self.__model__)
    elif self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_SD]):
        return StableDiffusionPipeline.from_pretrained(self.__model__,
            torch_dtype=torch.float16)
    elif self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_LDM]):
        pass
    elif self.__model__ in self.filter_mdid_by(archs=[ModelDataset.MD_NCSN,
        ModelDataset.MD_NCSNPP]):
        pass
    else:
        raise ValueError(f'Model, {self.__model__}, is not supported.')
