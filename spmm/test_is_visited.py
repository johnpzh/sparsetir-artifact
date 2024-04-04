
class Config:
    def __init__(self):
        self.num_parts = 0
        self.max_bucket_width = 0
        self.exe_time = 0.0
    def __init__(self, num_parts, max_bucket_width):
        self.num_parts = num_parts
        self.max_bucket_width = max_bucket_width
    def __eq__(self, other): 
        return self.num_parts == other.num_parts and \
                self.max_bucket_width == other.max_bucket_width
    

if __name__ == "__main__":
    a = Config(1, 1)
    queue = [a]
    b = Config(2, 2)
    print(F"b in queue: {b in queue}")
    queue.append(b)
    print(F"b in queue: {b in queue}")

    c = Config(1, 1)
    print(F"c in queue: {c in queue}")
