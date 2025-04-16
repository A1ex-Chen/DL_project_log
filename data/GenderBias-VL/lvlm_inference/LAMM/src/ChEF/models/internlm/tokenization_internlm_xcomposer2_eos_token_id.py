@property
def eos_token_id(self) ->Optional[int]:
    return self.sp_model.eos_id()
