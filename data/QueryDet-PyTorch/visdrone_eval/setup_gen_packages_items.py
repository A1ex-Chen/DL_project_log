def gen_packages_items():
    if exists(require_fpath):
        for info in parse_require_file(require_fpath):
            parts = [info['package']]
            if with_version and 'version' in info:
                parts.extend(info['version'])
            if not sys.version.startswith('3.4'):
                platform_deps = info.get('platform_deps')
                if platform_deps is not None:
                    parts.append(';' + platform_deps)
            item = ''.join(parts)
            yield item
