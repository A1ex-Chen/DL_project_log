def test_create_dummy_object(self):
    dummy_constant = create_dummy_object('CONSTANT', "'torch'")
    self.assertEqual(dummy_constant, '\nCONSTANT = None\n')
    dummy_function = create_dummy_object('function', "'torch'")
    self.assertEqual(dummy_function,
        """
def function(*args, **kwargs):
    requires_backends(function, 'torch')
"""
        )
    expected_dummy_class = """
class FakeClass(metaclass=DummyObject):
    _backends = 'torch'

    def __init__(self, *args, **kwargs):
        requires_backends(self, 'torch')

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, 'torch')

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, 'torch')
"""
    dummy_class = create_dummy_object('FakeClass', "'torch'")
    self.assertEqual(dummy_class, expected_dummy_class)
