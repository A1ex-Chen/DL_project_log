def test_list_move_three(self):
    n1 = self.list.add_to_front(1, 1)
    n2 = self.list.add_to_front(2, 2)
    self.list.add_to_front(3, 3)
    self.list.move_to_front(n1)
    self.assertEqual(self.list_to_array(), [(1, 1), (3, 3), (2, 2)])
    self.assertEqual(self.list_to_array(backward=True), [(2, 2), (3, 3), (1,
        1)])
    self.assertEqual(self.list.size, 3)
    self.list.move_to_front(n2)
    self.assertEqual(self.list_to_array(), [(2, 2), (1, 1), (3, 3)])
    self.assertEqual(self.list_to_array(backward=True), [(3, 3), (1, 1), (2,
        2)])
    self.assertEqual(self.list.size, 3)
    self.list.move_to_front(n2)
    self.assertEqual(self.list_to_array(), [(2, 2), (1, 1), (3, 3)])
    self.assertEqual(self.list_to_array(backward=True), [(3, 3), (1, 1), (2,
        2)])
    self.assertEqual(self.list.size, 3)
