def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """
    assert os.path.exists(file), 'file %s not found' % file
    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3
                ] == 'off', 'invalid OFF file %s' % file
            parts = lines[0][3:].split(' ')
            assert len(parts) == 3
            num_vertices = int(parts[0])
            assert num_vertices > 0
            num_faces = int(parts[1])
            assert num_faces > 0
            start_index = 1
        else:
            assert lines[0] == 'OFF' or lines[0
                ] == 'off', 'invalid OFF file %s' % file
            parts = lines[1].split(' ')
            assert len(parts) == 3
            num_vertices = int(parts[0])
            assert num_vertices > 0
            num_faces = int(parts[1])
            assert num_faces > 0
            start_index = 2
        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3
            vertices.append(vertex)
        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (
                    lines[start_index + num_vertices + i], file)
            face = [int(index) for index in face]
            assert face[0] == len(face
                ) - 1, 'face should have %d vertices but as %d (%s)' % (face
                [0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (
                    index, num_vertices, file)
            assert len(face) > 1
            faces.append(face)
        return vertices, faces
