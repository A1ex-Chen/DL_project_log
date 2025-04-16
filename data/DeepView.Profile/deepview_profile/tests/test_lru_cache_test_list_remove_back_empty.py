def test_list_remove_back_empty(self):
    removed = self.list.remove_back()
    self.assertEqual(self.list.size, 0)
    self.assertIsNone(removed)
    self.assertEqual(self.list_to_array(), [])
