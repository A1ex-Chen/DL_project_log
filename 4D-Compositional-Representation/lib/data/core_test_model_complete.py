def test_model_complete(self, category, model):
    """ Tests if model is complete.

        Args:
            model (str): modelname
        """
    model_path = os.path.join(self.dataset_folder, category, model)
    files = os.listdir(model_path)
    for field_name, field in self.fields.items():
        if not field.check_complete(files):
            logger.warn('Field "%s" is incomplete: %s' % (field_name,
                model_path))
            return False
    return True
