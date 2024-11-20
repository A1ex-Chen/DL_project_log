def test_print(self):
    sot = SingleObjectTracker()
    sot.update(detection=det0)
    sot.update(detection=det1)
    sot.update(detection=None)
    sot.update(detection=None)
    sot.update(detection=None)
    with captured_stdout() as stdout:
        print(sot)
    self.assertEqual(stdout.getvalue(),
        """(box: [20, 20, 3, 3], score: 0.75, class_id: 0, staleness: 3.00)
"""
        )
