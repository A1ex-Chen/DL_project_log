def _get_elem_from_tree(tree, tag):
    return tree.getElementsByTagName(tag)[0].firstChild.data
