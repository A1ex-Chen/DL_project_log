def read_pitch_keypoints(xmlfile: str, annot_type: str) ->tuple[NDArray[np.
    float64], NDArray[np.float64]]:
    """Read pitch keypoints from xml file.

    Args:
        xmlfile (str): path to xml file.
        annot_type (str): type of annotation. Either 'pitch' or 'video'.

    Raises:
        ValueError: if annotation type is not 'pitch' or 'video'.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: pitch keypoints and video keypoints.

    """
    tree = ElementTree.parse(xmlfile)
    root = tree.getroot()
    src = []
    dst = []
    if annot_type == 'video':
        for child in root:
            for c in child:
                d = c.attrib
                if d != {}:
                    dst.append(eval(d['label']))
                    src.append(eval(d['points']))
    elif annot_type == 'frame':
        for child in root:
            d = child.attrib
            if d != {}:
                dst.append(eval(d['label']))
                src.append(eval(child[0].attrib['points']))
    else:
        raise ValueError('Annotation type must be `video` or `frame`.')
    src = np.asarray(src)
    dst = np.asarray(dst)
    assert src.size != 0, 'No keypoints found in XML file.'
    return src, dst
