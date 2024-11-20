def test_list_move_two(self):
    n1 = self.list.add_to_front(1, 1)
    self.list.add_to_front(2, 2)
    self.assertEqual(self.list_to_array(), [(2, 2), (1, 1)])
    self.list.move_to_front(n1)
    self.assertEqual(self.list_to_array(), [(1, 1), (2, 2)])
