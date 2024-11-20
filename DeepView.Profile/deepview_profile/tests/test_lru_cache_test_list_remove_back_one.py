def test_list_remove_back_one(self):
    n1 = self.list.add_to_front(1, 1)
    self.assertEqual(self.list_to_array(), [(1, 1)])
    removed = self.list.remove_back()
    self.assertEqual(n1, removed)
    self.assertEqual(self.list.size, 0)
    self.assertEqual(self.list_to_array(), [])
