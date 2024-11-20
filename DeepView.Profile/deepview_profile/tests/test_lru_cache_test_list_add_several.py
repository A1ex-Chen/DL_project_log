def test_list_add_several(self):
    self.list.add_to_front(1, 1)
    self.list.add_to_front(2, 2)
    self.list.add_to_front(3, 3)
    self.assertEqual(self.list.size, 3)
    self.assertNotEqual(self.list.front, self.list.back)
    self.assertEqual(self.list_to_array(), [(3, 3), (2, 2), (1, 1)])
