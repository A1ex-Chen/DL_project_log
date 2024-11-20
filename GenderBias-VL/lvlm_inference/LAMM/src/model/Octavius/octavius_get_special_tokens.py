def get_special_tokens(self):
    tokens = []
    for modality in self.vision_type:
        sov = VISION_TAGS['sov'][modality]
        eov = VISION_TAGS['eov'][modality]
        tokens.extend([sov, eov])
        print(f'Add VISION TAG ("{sov}" and "{eov}") for modality {modality}.')
    return tokens
