def _downByThread(self, start, end):
    for i in range(start, end + 1):
        self._getPicture(self.m_lstInfoItems[i])
