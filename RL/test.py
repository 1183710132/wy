class Example:
    def __init__(self, value):
        self._value = value

    @property
    def doubled_value(self):
        print("Calculating doubled value...")
        return self._value * 2

obj = Example(10)
print(obj.doubled_value)  # 输出: "Calculating doubled value..." 然后输出 20
print(obj.doubled_value)  # 再次输出: "Calculating doubled value..." 然后输出 20
