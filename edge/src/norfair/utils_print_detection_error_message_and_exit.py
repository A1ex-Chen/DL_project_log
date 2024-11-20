def print_detection_error_message_and_exit(points):
    print('\n[red]INPUT ERROR:[/red]')
    print(
        f'Each `Detection` object should have a property `points` of shape (num_of_points_to_track, 2), not {points.shape}. Check your `Detection` list creation code.'
        )
    print('You can read the documentation for the `Detection` class here:')
    print(
        'https://tryolabs.github.io/norfair/reference/tracker/#norfair.tracker.Detection\n'
        )
    exit()
