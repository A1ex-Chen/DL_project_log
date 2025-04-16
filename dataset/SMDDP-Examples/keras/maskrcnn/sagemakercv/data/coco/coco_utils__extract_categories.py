def _extract_categories(annotations):
    """Extract categories from annotations."""
    categories = {}
    for anno in annotations:
        category_id = int(anno['category_id'])
        categories[category_id] = {'id': category_id}
    return list(categories.values())
