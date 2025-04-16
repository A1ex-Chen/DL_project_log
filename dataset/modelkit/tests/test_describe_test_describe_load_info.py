def test_describe_load_info():


    class top(Model[str, str]):
        CONFIGURATIONS = {'top': {'model_dependencies': ['right', 'left']}}

        def _predict(self, item):
            return item


    class right(Model[str, str]):
        CONFIGURATIONS = {'right': {'model_dependencies': ['right_dep',
            'join_dep']}}

        def _predict(self, item):
            return item


    class left(Model[str, str]):
        CONFIGURATIONS = {'left': {'model_dependencies': ['join_dep']}}

        def _predict(self, item):
            return item


    class right_dep(Model[str, str]):
        CONFIGURATIONS = {'right_dep': {}}

        def _predict(self, item):
            return item


    class join_dep(Model[str, str]):
        CONFIGURATIONS = {'join_dep': {}}

        def _predict(self, item):
            return item
    console = Console(no_color=True, force_terminal=False, width=130)
    library = ModelLibrary(models=[top, right, left, join_dep, right_dep])
    for m in ['top', 'right', 'left', 'join_dep', 'right_dep']:
        library.get(m)._load_time = 0.1
        library.get(m)._load_memory_increment = 2
    load_info_top = {}
    add_dependencies_load_info(load_info_top, library.get('top'))
    assert load_info_top == {'right': {'time': 0.1, 'memory_increment': 2},
        'left': {'time': 0.1, 'memory_increment': 2}, 'join_dep': {'time': 
        0.1, 'memory_increment': 2}, 'right_dep': {'time': 0.1,
        'memory_increment': 2}}
    load_info_right = {}
    add_dependencies_load_info(load_info_right, library.get('right'))
    assert load_info_right == {'join_dep': {'time': 0.1, 'memory_increment':
        2}, 'right_dep': {'time': 0.1, 'memory_increment': 2}}
    load_info_join_dep = {}
    add_dependencies_load_info(load_info_join_dep, library.get('join_dep'))
    assert load_info_join_dep == {}
    if platform.system() == 'Windows' or sys.version_info[:2] < (3, 11):
        return
    with console.capture() as capture:
        console.print('join_dep describe:')
        console.print(library.get('join_dep').describe())
        console.print()
        console.print('top describe:')
        console.print(library.get('top').describe())
    r = ReferenceText(os.path.join(TEST_DIR, 'testdata'))
    r.assert_equal('test_describe_load_info.txt', capture.get())
