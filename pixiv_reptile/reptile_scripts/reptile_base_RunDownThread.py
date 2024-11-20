def RunDownThread(self):
    num_batch = len(self.m_lstInfoItems)
    num_thread = (defines.NUM_THREAD if num_batch > defines.NUM_THREAD else
        num_batch)
    elements = num_batch // num_thread
    remaining_elements = num_batch % num_thread
    for i in range(num_thread):
        start = i * elements + min(i, remaining_elements)
        end = start + elements + (1 if i < remaining_elements else 0) - 1
        thread = threading.Thread(target=self._downByThread, args=(start, end))
        self.m_lstThread.append(thread)
        thread.start()
    for thread in self.m_lstThread:
        thread.join()
    print('清空线程列表')
    del self.m_lstThread[:]
