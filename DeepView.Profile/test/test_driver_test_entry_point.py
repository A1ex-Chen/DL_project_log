@pytest.mark.parametrize('test_name, entry_point', tests)
def test_entry_point(test_name, entry_point):
    print(f'Testing {entry_point}')
    folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))
        ) + '/tests_results'
    os.makedirs(folder, exist_ok=True)
    stdout_fd = open(os.path.join(folder,
        f'{test_name}_interactive_output.log'), 'w')
    stderr_fd = open(os.path.join(folder,
        f'{test_name}_interactive_w_debug_output.log'), 'w')
    context = BackendContext(entry_point, stdout_fd=stdout_fd, stderr_fd=
        stderr_fd)
    context.spawn_process()
    analysis_messages = list()
    for reps in range(REPS):
        sess = DeepviewSession()
        while context.state == 0:
            pass
        sess.connect('localhost', 60120)
        sess.send_initialize_request(entry_point)
        sess.send_analysis_request()
        while context.alive() and sess.alive() and len(sess.received_messages
            ) < NUM_EXPECTED_MESSAGES:
            pass
        sess.cleanup()
        analysis_messages.extend(sess.received_messages)
        assert len(sess.received_messages
            ) == NUM_EXPECTED_MESSAGES, f'Run {reps}: Expected to receive {NUM_EXPECTED_MESSAGES} got {len(sess.received_messages)} (did the process terminate prematurely?)'
    context.terminate()
    with open(os.path.join(folder, f'{test_name}_analysis.pkl'), 'wb') as fp:
        pickle.dump(list(map(MessageToDict, analysis_messages)), fp)
    package_dict = get_config_name()
    with open(os.path.join(folder, 'package-list.txt'), 'w') as f:
        for k, v in package_dict.items():
            f.write(f'{k}={v}\n')
    stdout_fd.close()
    stderr_fd.close()
