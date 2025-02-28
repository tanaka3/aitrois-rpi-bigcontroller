class CommandParser:
    def __init__(self):
        self.RESET = False
        self.UP = False
        self.DOWN = False
        self.LEFT = False
        self.RIGHT = False
        self.A = False
        self.B = False
        self.START = False
        self.SELECT = False
        self.reset()

    def reset(self):
        self.RESET = False
        self.UP = False
        self.DOWN = False
        self.LEFT = False
        self.RIGHT = False
        self.A = False
        self.B = False
        self.START = False
        self.SELECT = False

    def parse(self, command: int):
        self.SELECT = (command & (1 << 0)) != 0
        self.START = (command & (1 << 1)) != 0
        self.A = (command & (1 << 2)) != 0
        self.B = (command & (1 << 3)) != 0
        self.RIGHT = (command & (1 << 4)) != 0
        self.LEFT = (command & (1 << 5)) != 0
        self.DOWN = (command & (1 << 6)) != 0
        self.UP = (command & (1 << 7)) != 0
        self.RESET = (command & (1 << 8)) != 0

    def print_status(self):
        print("===============================")
        print(f"RESET: {self.RESET}")
        print(f"UP: {self.UP}")
        print(f"DOWN: {self.DOWN}")
        print(f"LEFT: {self.LEFT}")
        print(f"RIGHT: {self.RIGHT}")
        print(f"A: {self.A}")
        print(f"B: {self.B}")
        print(f"START: {self.START}")
        print(f"SELECT: {self.SELECT}")

    @staticmethod
    def is_valid_command(data: str) -> bool:
        if data.startswith("D:"):
            number_str = data[2:]
            return number_str.isdigit()
        return False

    def generate_command(self) -> int:
        command = 0
        if self.RESET:
            command |= (1 << 8)
        if self.UP:
            command |= (1 << 7)
        if self.DOWN:
            command |= (1 << 6)
        if self.LEFT:
            command |= (1 << 5)
        if self.RIGHT:
            command |= (1 << 4)
        if self.B:
            command |= (1 << 3)
        if self.A:
            command |= (1 << 2)
        if self.START:
            command |= (1 << 1)
        if self.SELECT:
            command |= (1 << 0)
        return command

    def send_command(self, serial_port):
        command = self.generate_command()
        serial_port.write(f"D:{command}\n".encode())
