def _predict(self, item):
    """Run "dynamic_model" twice with different argument to simulate
        python caching.

        """
    time.sleep(0.3)
    item = self.dynamic_model(item, call_x=True)
    item = self.dynamic_model(item, call_x=False)
    return item
