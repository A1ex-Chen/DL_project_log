def process_target(self, raw_conv: List[Dict[str, Any]], target: Dict[str,
    Any], multimage_mode=False) ->Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
        convert target placeholder to actual information in raw_conv.
            e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
        """
    return self.process_func['target'](raw_conv, target, self.preprocessor,
        multimage_mode=multimage_mode)
