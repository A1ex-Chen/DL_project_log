def __eq__(self, other):
    return (self.major == other.major and self.minor == other.minor and 
        self.patch == other.patch)
