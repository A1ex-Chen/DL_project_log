def search(self, variable_name):
    for line in self.lines:
        if line[:len(variable_name)] == variable_name:
            result = line[len(variable_name) + 1:]
            break
    else:
        raise ValueError(f"Couldn't find '{variable_name}' in {self.path}")
    if result.isdigit():
        return int(result)
    else:
        return result
