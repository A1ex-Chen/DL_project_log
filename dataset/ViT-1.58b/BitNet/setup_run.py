def run(self):
    if FORCE_BUILD:
        return super().run()
    wheel_url, wheel_filename = get_wheel_url()
    print('Guessing wheel URL: ', wheel_url)
    try:
        urllib.request.urlretrieve(wheel_url, wheel_filename)
        if not os.path.exists(self.dist_dir):
            os.makedirs(self.dist_dir)
        impl_tag, abi_tag, plat_tag = self.get_tag()
        archive_basename = (
            f'{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}')
        wheel_path = os.path.join(self.dist_dir, archive_basename + '.whl')
        print('Raw wheel path', wheel_path)
        shutil.move(wheel_filename, wheel_path)
    except urllib.error.HTTPError:
        print('Precompiled wheel not found. Building from source...')
        super().run()
