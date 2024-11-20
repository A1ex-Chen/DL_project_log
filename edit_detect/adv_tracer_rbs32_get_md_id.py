@staticmethod
def get_md_id(md_name: str):
    if md_name == ModelSched.MD_NAME_CD_L2_IMAGENET64:
        return ModelSched.MD_ID_CD_L2_IMAGENET64
    elif md_name == ModelSched.MD_NAME_CD_LPIPS_IMAGENET64:
        return ModelSched.MD_ID_CD_LPIPS_IMAGENET64
    elif md_name == ModelSched.MD_NAME_CT_IMAGENET64:
        return ModelSched.MD_ID_CT_IMAGENET64
    elif md_name == ModelSched.MD_NAME_CD_L2_BEDROOM256:
        return ModelSched.MD_ID_CD_L2_BEDROOM256
    elif md_name == ModelSched.MD_NAME_CD_LPIPS_BEDROOM256:
        return ModelSched.MD_ID_CD_LPIPS_BEDROOM256
    elif md_name == ModelSched.MD_NAME_CT_BEDROOM256:
        return ModelSched.MD_ID_CT_BEDROOM256
    else:
        raise ValueError(f"Arguement md_name, {md_name}, doesn't support ")
