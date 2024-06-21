
class Member:
    def __init__(self):
        self.age = 1
    
    def __str__(self) -> str:
        return F"Member(age: {self.age})"

if __name__ == "__main__":
    d = {
        1: Member(),
        2: Member(),
        4: Member()
    }

    for key in d.keys():
        if key == 4:
            continue
        new_key = 2 * key
        d[new_key].age += d[key].age
    
    
    # Print
    for key, val in d.items():
        print(F"key: {key} val: {val}")
