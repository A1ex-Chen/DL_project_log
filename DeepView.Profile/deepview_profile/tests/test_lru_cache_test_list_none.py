def test_list_none(self):
    self.assertIsNone(self.list.front)
    self.assertIsNone(self.list.back)
    self.assertEqual(self.list.size, 0)
    node = self.list.add_to_front(1, 1)
    self.assertIsNone(node.next)
    self.assertIsNone(node.prev)
    self.assertEqual(self.list.size, 1)
