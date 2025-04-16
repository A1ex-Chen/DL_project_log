def close(self):
    if self.requests_session:
        return self.requests_session.close()
