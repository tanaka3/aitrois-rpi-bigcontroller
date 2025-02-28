#pragma once

class CommandParser {

  public:
    bool RESET = false;
    bool UP = false;
    bool DOWN = false;
    bool LEFT = false;
    bool RIGHT = false;
    bool A = false;
    bool B = false;
    bool START = false;
    bool SELECT = false;

    CommandParser() {
      reset();
    }

    void reset(){
      RESET = false;
      UP = false;
      DOWN = false;
      LEFT = false;
      RIGHT = false;
      A = false;
      B = false;
      START = false;
      SELECT = false;
    }

    void parse(int command) {

      SELECT =(command & (1 << 0)) != 0; //最下位ビット
      START = (command & (1 << 1)) != 0;
      A =     (command & (1 << 2)) != 0;
      B =     (command & (1 << 3)) != 0;
      RIGHT = (command & (1 << 4)) != 0;
      LEFT =  (command & (1 << 5)) != 0;
      DOWN =  (command & (1 << 6)) != 0;
      UP =    (command & (1 << 7)) != 0;
      RESET = (command & (1 << 8)) != 0; //最上位ビット

    }

    void printStatus() {
      Serial.println("===============================");
      Serial.print("RESET: "); Serial.println(RESET);
      Serial.print("UP: "); Serial.println(UP);
      Serial.print("DOWN: "); Serial.println(DOWN);
      Serial.print("LEFT: "); Serial.println(LEFT);
      Serial.print("RIGHT: "); Serial.println(RIGHT);
      Serial.print("A: "); Serial.println(A);
      Serial.print("B: "); Serial.println(B);
      Serial.print("START: "); Serial.println(START);      
      Serial.print("SELECT: "); Serial.println(SELECT);
    }

    static bool isValidCommand(String data) {
      if (data.startsWith("D:")) {
        String numberStr = data.substring(2);
        int command = numberStr.toInt();
        return command != 0 || numberStr == "0";
      }
      return false;
    }

    int generateCommand() {
      int command = 0;
      
      if (RESET)  command |= (1 << 8);
      if (UP)     command |= (1 << 7);   
      if (DOWN)   command |= (1 << 6); 
      if (LEFT)   command |= (1 << 5); 
      if (RIGHT)  command |= (1 << 4);
      if (B)      command |= (1 << 3);
      if (A)      command |= (1 << 2);
      if (START)  command |= (1 << 1);
      if (SELECT) command |= (1 << 0);

      return command;
    }

    void sendCommand(Stream &serial) {
      int command = generateCommand();
      serial.print("D:");
      serial.println(command);  // "D:" とコマンドを送信
    }    
};