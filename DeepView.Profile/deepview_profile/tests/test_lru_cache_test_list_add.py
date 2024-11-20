def test_list_add(self):
    self.list.add_to_front('hello', 'world')
    self.assertEqual(self.list.size, 1)
    self.assertEqual(self.list.front, self.list.back)
    self.assertIsNone(self.list.front.next)
    self.assertIsNone(self.list.front.prev)
    self.assertEqual(self.list.front.key, 'hello')
    self.assertEqual(self.list.front.value, 'world')
    self.list.add_to_front('hello2', 'world2')
    self.assertEqual(self.list.size, 2)
    self.assertEqual(self.list_to_array(backward=True), [('hello', 'world'),
        ('hello2', 'world2')])
