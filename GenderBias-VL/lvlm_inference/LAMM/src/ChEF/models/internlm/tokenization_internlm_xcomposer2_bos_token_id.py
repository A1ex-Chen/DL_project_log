@property
def bos_token_id(self) ->Optional[int]:
    return self.sp_model.bos_id()
