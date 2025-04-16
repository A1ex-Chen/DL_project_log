def delete_object(self, object_name):
    object_path = os.path.join(self.bucket, *object_name.split('/'))
    if os.path.exists(object_path):
        os.remove(object_path)
