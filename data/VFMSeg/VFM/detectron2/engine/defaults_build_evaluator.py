@classmethod
def build_evaluator(cls, cfg, dataset_name):
    """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
    raise NotImplementedError(
        """
If you want DefaultTrainer to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
"""
        )
