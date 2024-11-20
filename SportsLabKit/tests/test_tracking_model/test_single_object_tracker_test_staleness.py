def test_staleness(self):
    sot = SingleObjectTracker(max_staleness=2)
    sot.update(detection=det0)
    sot.update(detection=det1)
    sot.update(detection=None)
    sot.update(detection=None)
    sot.update(detection=None)
    self.assertEqual(sot.steps_positive, 2)
    self.assertEqual(sot.steps_alive, 5)
    self.assertEqual(sot.staleness, 3)
    self.assertTrue(sot.is_stale)