// https://wiki.seeedstudio.com/XIAO-RP2040-with-Arduino/
#include <Adafruit_NeoPixel.h>
#include "switch_tinyusb_custom.h"
#include "CommandParser.hpp"

#define NUMPIXELS 1
#define NEO_PWR 11
#define NEOPIX 12

Adafruit_NeoPixel pixels(NUMPIXELS, NEOPIX, NEO_GRB + NEO_KHZ800);

Adafruit_USBD_HID G_usb_hid;
NSGamepad Gamepad(&G_usb_hid);
HID_NSGamepadReport_Data_t beforeData;

static CommandParser commandParser;

bool resetFlg = false;

void setup() {

  pinMode(NEO_PWR,OUTPUT);  
  digitalWrite(NEO_PWR, HIGH);
  
  pixels.begin();
  pixels.setBrightness(64);

  pixels.setPixelColor(0, pixels.Color(255, 0, 0));  
  pixels.show(); 

  Gamepad.begin();
  while( !USBDevice.mounted() ) delay(1);
  
  pixels.setPixelColor(0, pixels.Color(0, 255, 0));  
  pixels.show(); 
}

void loop() {
  Gamepad.reset();

  if(commandParser.START){
      if(resetFlg){
        return;
      }

      if(commandParser.SELECT){
        resetFlg = true;
        resetControll();
      }
      else{
        Gamepad.press(NSButton_Plus);
      }

  }
  else{
    resetFlg = false;
    
    if(commandParser.SELECT){
      Gamepad.press(NSButton_Minus);         
    }

    if(commandParser.A){
      Gamepad.press(NSButton_A);         
    }

    if(commandParser.B){
      Gamepad.press(NSButton_B);         
    }

    Gamepad.dPad(commandParser.UP, commandParser.DOWN, 
                  commandParser.LEFT, commandParser.RIGHT);
  }

  // 前回と入力が同じ場合は、ボタン操作を送信しない
  if(!Gamepad.compareTo(beforeData)){
    Gamepad.SendReport();
    beforeData = Gamepad.getReportData();
    //Serial.println("PUSH");
  }  
}

void setup1(){
  Serial.begin(115200);
  Serial1.begin(115200);
}

void loop1(){
  if (Serial1.available() > 0) {
    String receivedData = Serial1.readStringUntil('\n');

    Serial.println(receivedData);
    
    if(CommandParser::isValidCommand(receivedData)) {
      String numberStr = receivedData.substring(2);
      int command = numberStr.toInt();
      commandParser.parse(command);
      //commandParser.printStatus();

      // Serial.print(receivedData);
      // Serial.print(":");
      // Serial.print(command);
      // Serial.print(":");
      // Serial.println(commandParser.START);
     
    }
    else{
      commandParser.reset();
    }

  }
}

/**
 * @brief ファミコンのリセット操作 
 * ZL＋ZR押してメニューのリセットを選ぶ操作
 */
void resetControll(){

  Gamepad.reset();
  Gamepad.press(NSButton_LeftThrottle);
  Gamepad.press(NSButton_RightThrottle);  
  Gamepad.SendReport();
  delay(100);

  for(int i=0; i<3; i++){
    Gamepad.reset();
    Gamepad.dPad(true, false, false, false);
    Gamepad.SendReport();
    delay(150);
    Gamepad.reset();
    Gamepad.SendReport();
    delay(150);
  }
  
  Gamepad.reset();
  Gamepad.press(NSButton_A);
  Gamepad.SendReport();
};
