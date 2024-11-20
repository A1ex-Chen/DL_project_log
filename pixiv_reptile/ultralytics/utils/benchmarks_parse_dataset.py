def parse_dataset(self, ds_link_txt='datasets_links.txt'):
    """
        Parse dataset links and downloads datasets.

        Args:
            ds_link_txt (str): Path to dataset_links file.
        """
    (shutil.rmtree('rf-100'), os.mkdir('rf-100')) if os.path.exists('rf-100'
        ) else os.mkdir('rf-100')
    os.chdir('rf-100')
    os.mkdir('ultralytics-benchmarks')
    safe_download(
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/datasets_links.txt'
        )
    with open(ds_link_txt, 'r') as file:
        for line in file:
            try:
                _, url, workspace, project, version = re.split('/+', line.
                    strip())
                self.ds_names.append(project)
                proj_version = f'{project}-{version}'
                if not Path(proj_version).exists():
                    self.rf.workspace(workspace).project(project).version(
                        version).download('yolov8')
                else:
                    print('Dataset already downloaded.')
                self.ds_cfg_list.append(Path.cwd() / proj_version / 'data.yaml'
                    )
            except Exception:
                continue
    return self.ds_names, self.ds_cfg_list
