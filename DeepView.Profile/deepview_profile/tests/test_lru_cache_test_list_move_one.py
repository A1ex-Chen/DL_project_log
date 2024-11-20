def test_list_move_one(self):
    n1 = self.list.add_to_front(1, 1)
    self.assertEqual(self.list_to_array(), [(1, 1)])
    self.list.move_to_front(n1)
    self.assertEqual(self.list_to_array(), [(1, 1)])
    self.assertEqual(self.list_to_array(backward=True), [(1, 1)])
    self.assertEqual(self.list.size, 1)
