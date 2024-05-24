class Test:
    def __init__(self, name):
        self.name = name
        self.ds = []

    def get(self, string_name):
        fmt = "test name instance [{}] is getting you {}".format(self.name, string_name)
        return fmt

    def add(self, string_name):
        self.ds.append(string_name)
        return string_name

    def get_all(self):
        fmt = ""
        for s in self.ds:
            fmt = fmt + s
        fmt = "test name instance [{}] is getting you {}".format(self.name, fmt)
        return fmt
