class TimeManager:
    def __init__(self, timestamps):
        self.timestamps = timestamps
        self.it = iter(timestamps)
        self.now = -1

    def update_timestamp(self):
        self.now = next(self.it)

    def get_timestamp(self):
        return self.now