def maybe_download_raw_dataset(self):
    folder_path = self._get_rawdata_folder_path()
    if folder_path.is_dir() and all(folder_path.joinpath(filename).is_file(
        ) for filename in self.all_raw_file_names()):
        print('Raw data already exists. Skip downloading')
        return
    print("Raw file doesn't exist. Downloading...")
    if self.is_zipfile():
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.zip')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        unzip(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print()
    else:
        tmproot = Path(tempfile.mkdtemp())
        tmpfile = tmproot.joinpath('file')
        download(self.url(), tmpfile)
        folder_path.mkdir(parents=True)
        shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
        shutil.rmtree(tmproot)
        print()
