def export_off(vertices, triangles, filename):
    """
    Exports a mesh in the (.off) format.
    """
    with open(filename, 'w') as fh:
        fh.write('OFF\n')
        fh.write('{} {} 0\n'.format(len(vertices), len(triangles)))
        for v in vertices:
            fh.write('{} {} {}\n'.format(*v))
        for f in triangles:
            fh.write('3 {} {} {}\n'.format(*f))
