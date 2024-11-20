def _upload_folder(self, working_dir: Union[str, os.PathLike], repo_id: str,
    token: Optional[str]=None, commit_message: Optional[str]=None,
    create_pr: bool=False):
    """
        Uploads all files in `working_dir` to `repo_id`.
        """
    if commit_message is None:
        if 'Model' in self.__class__.__name__:
            commit_message = 'Upload model'
        elif 'Scheduler' in self.__class__.__name__:
            commit_message = 'Upload scheduler'
        else:
            commit_message = f'Upload {self.__class__.__name__}'
    logger.info(f'Uploading the files of {working_dir} to {repo_id}.')
    return upload_folder(repo_id=repo_id, folder_path=working_dir, token=
        token, commit_message=commit_message, create_pr=create_pr)
