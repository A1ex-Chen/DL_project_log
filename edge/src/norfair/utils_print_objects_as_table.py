def print_objects_as_table(tracked_objects: Sequence):
    """Used for helping in debugging"""
    print()
    console = Console()
    table = Table(show_header=True, header_style='bold magenta')
    table.add_column('Id', style='yellow', justify='center')
    table.add_column('Age', justify='right')
    table.add_column('Hit Counter', justify='right')
    table.add_column('Last distance', justify='right')
    table.add_column('Init Id', justify='center')
    for obj in tracked_objects:
        table.add_row(str(obj.id), str(obj.age), str(obj.hit_counter),
            f'{obj.last_distance:.4f}', str(obj.initializing_id))
    console.print(table)
