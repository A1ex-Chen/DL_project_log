def dict(self):
    if len(self.get_images()) > 0:
        return {'system': self.system, 'roles': self.roles, 'messages': [[x,
            y[0] if type(y) is tuple else y] for x, y in self.messages],
            'offset': self.offset, 'sep': self.sep, 'sep2': self.sep2}
    return {'system': self.system, 'roles': self.roles, 'messages': self.
        messages, 'offset': self.offset, 'sep': self.sep, 'sep2': self.sep2}
