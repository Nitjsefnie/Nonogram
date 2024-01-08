class StringBuilder:
    def __init__(self):
        self.values = []

    def add(self, *args, end="\n", sep=" "):
        for value in args:
            if value is not None:
                if isinstance(value, str):
                    self.values.append(value)
                else:
                    self.values.add(str(value))
            self.values.append(sep)
        self.values.pop()
        self.values.append(end)
        return self

    def build(self):
        return ''.join(self.values)
