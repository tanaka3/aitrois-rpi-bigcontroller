import serial

class ButtonDebouncer:
    """
    Determine if it's pressed

    Attributes:
        consecutive: The number of consecutive observations of the opposite state
        state: Current stable state (initial state is False)
        threshold_true (int): If the current state is False, 
                                the number of consecutive True observations required to change the state to True.
        threshold_false (int): If currently True, 
                                the number of consecutive False observations required to change the state to False.
    """    
    def __init__(self, threshold_true=3, threshold_false=3):
        self.threshold_true = threshold_true
        self.threshold_false = threshold_false
        self.state = False
        self.consecutive = 0


    def update(self, raw_state: bool):
        """
        Update the stable state (state) based on the raw input state (raw_state).
        If a value different from the current state is observed consecutively, 
        the state will be reversed when the set threshold is reached.
        
        Args:
            raw_state (bool): Raw input state from sensors etc.
        """
        if raw_state == self.state:
            self.consecutive = 0
        else:
            self.consecutive += 1

            if not self.state and raw_state:
                if self.consecutive >= self.threshold_true:
                    self.state = True
                    self.consecutive = 0
            elif self.state and not raw_state:
                if self.consecutive >= self.threshold_false:
                    self.state = False
                    self.consecutive = 0

    def is_pressed(self) -> bool:
        """
        If the counter is equal to or greater than the threshold, the button is determined to be "pressed."
        """
        return self.state   
    
class FamicomControllerState:
    """
    A class that manages the ON/OFF state of each button on a Famicom controller.

    Attributes:
        RESET (bool): リSpecial key for resetting Famicom games on Nintendo Switch
        UP (bool): The state of the up button.
        DOWN (bool): The state of the down button.
        LEFT (bool): The state of the left button.
        RIGHT (bool): The state of the right button.
        A (bool): The state of the a button.
        B (bool): The state of the b button.
        START (bool): The state of the start button.
        SELECT (bool): The state of the select button.
    """
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

    def parse_command(self, command: int):
        self.SELECT = (command & (1 << 0)) != 0
        self.START = (command & (1 << 1)) != 0
        self.A = (command & (1 << 2)) != 0
        self.B = (command & (1 << 3)) != 0
        self.RIGHT = (command & (1 << 4)) != 0
        self.LEFT = (command & (1 << 5)) != 0
        self.DOWN = (command & (1 << 6)) != 0
        self.UP = (command & (1 << 7)) != 0
        self.RESET = (command & (1 << 8)) != 0

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
        """
        Check only the format
        """
        if data.startswith("D:"):
            number_str = data[2:]
            return number_str.isdigit()
        return False


    def send_command(self, serial_port):
        command = self.generate_command()
        serial_port.write(f"D:{command}\n".encode())


class DebouncedFamicomControllerState:
    def __init__(self, default_threshold_true=3, default_threshold_false=3, button_thresholds=None):
        """
        default_threshold: すべてのボタンに適用する基本のフレーム数閾値
        button_thresholds: 特定のボタンに対して異なる閾値を指定する辞書（例: {'A': 5}）
        """
        # ボタン一覧
        self.buttons = ['RESET', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']
        if button_thresholds is None:
            button_thresholds = {}

        # 各ボタンごとにデバウンサを初期化（Aボタンは特別な閾値を使うなど）
        self.debouncers = {}
        for button in self.buttons:
            thresholds = button_thresholds.get(button, (default_threshold_true, default_threshold_false))
            self.debouncers[button] = ButtonDebouncer(threshold_true=thresholds[0], threshold_false=thresholds[1])

    def update(self, raw_state: FamicomControllerState):
        """
        It receives raw command information for each frame and performs debounce processing for each button.
        """
        for button in self.buttons:
            raw_value = getattr(raw_state, button)
            self.debouncers[button].update(raw_value)

    def get_debounced_state(self) -> FamicomControllerState:
        """
        The command state after debounce processing is stored in a new FamicomControllerState instance and returned.
        """
        debounced_state = FamicomControllerState()
        for button in self.buttons:
            setattr(debounced_state, button, self.debouncers[button].is_pressed())
        return debounced_state

class FamicomControllerSender:
    """
    pip install pyserial
    """
    def __init__(self, port):
        """
        Args:
            port (str): port name.
        """        
        try:
            self.serial_port = serial.Serial(port, baudrate=115200, timeout=1)
        except serial.SerialException as e:
             print(f"Failed to open serial port: {e}")
             

    def send_command(self, state: FamicomControllerState):

        if not self.serial_port or not self.serial_port.is_open:
            print("The serial port is not open so commands cannot be sent.")
            return
                
        command = state.generate_command()        
        self.serial_port.write(f"D:{command}\n".encode())
