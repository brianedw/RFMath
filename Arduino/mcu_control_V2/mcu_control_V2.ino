
// COM8 -> 1
// COM4 -> 2
#define idNum 1
  
#include <SoftwareSerial.h>

#define OUTPUT_MIN 0
#define OUTPUT_MAX 0x03ff

// VGA
// targ0
#define writePin0   9 
#define writePin0F  6 
#define readPin0    A1

// Phase Shifter
// targ1
#define writePin1   10
#define writePin1F  5
#define readPin1    A0

// Communicatiosn
// On input, pins read  [Gnd, Rx,   ]
//                      [   , 11, 13]
#define serRxIn     11

// On output, pins read [Gnd, Tx,   ]
//                      [   ,  2,  3]
#define serTxOut    2

// Tasks
#define blink_task  1
#define set_task    2

int targ0, targ1, valRead0, valRead1, valOut0, valOut1, dValOut0, dValOut1, targ0New, targ1New;
long lastUpdateTime, now, nowSetting;

unsigned long secs, secsOld;

SoftwareSerial otherSerial =  SoftwareSerial(serRxIn, serTxOut); // Rx, Tx

void setupSerials() {
  Serial.begin(9600);
  otherSerial.begin(9600);
  Serial.setTimeout(100);
  otherSerial.setTimeout(100);
  delay(100);
  pinMode(serRxIn, INPUT);
  pinMode(serTxOut, OUTPUT);
}

/* 10-bit will have a range of up to 0x03ff (1023) */
void setupPWM10() {
  DDRB |= _BV(PB1) | _BV(PB2);                              /* set pins as outputs */
  TCCR1A = _BV(COM1A1) | _BV(COM1B1)  | _BV(WGM11);         /* non-inverting PWM */
  TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS10);             /* no prescaling */
  ICR1 = 0x03ff;                                            /* TOP counter value */
}

/* 10-bit version of analogWrite(). Works only on pins 9 and 10. */
void analogWrite10(uint8_t pin, uint16_t val)
{
  switch (pin) {
    case  9: OCR1A = val; break;
    case 10: OCR1B = val; break;
  }
}

void updateReadings(){
  valRead0 = analogRead(readPin0);
  valRead1 = analogRead(readPin1);
}

void checkGood(){
  if(abs(valRead0 - targ0)<=3 && abs(valRead1 - targ1)<=3){
    digitalWrite(LED_BUILTIN, LOW);
  } else {
    digitalWrite(LED_BUILTIN, HIGH);
  }
}

void fastSet(){
  pinMode(writePin0F, OUTPUT);
  pinMode(writePin1F, OUTPUT);
  analogWrite10(writePin0, targ0);
  analogWrite10(writePin1, targ1);
  analogWrite(writePin0F, (uint8_t)(targ0>>2));
  analogWrite(writePin1F, (uint8_t)(targ1>>2));
  delay(20);
  pinMode(writePin0F, INPUT);
  pinMode(writePin1F, INPUT);
}

void printAllReadings(){
  static unsigned long lastPrint = 0;
  if(millis() - lastPrint > 1000){
    lastPrint = millis();
    Serial.print(targ0);Serial.print(", ");Serial.println(valRead0);
    Serial.print(targ1);Serial.print(", ");Serial.println(valRead1);
    Serial.println();
  }

}

void blinkID(){
  for(int i=0; i<5000/1000; i++){
    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
  }
}

void flashWarning(){
  for(int i=0; i<3000/100; i++){
    digitalWrite(LED_BUILTIN, HIGH);
    delay(50);
    digitalWrite(LED_BUILTIN, LOW);
    delay(50);
  }
}

void clearSerial(Stream &ser){
  delay(10);
  while(ser.available()){
    ser.read();
  }
}

int manageInput(Stream &inSer, Stream &outSer) {
  /*
   * Input of the form of three ints:
   * "idDest task [param1] [param2]"
   */
  if(inSer.available() == 0){ // Nothing to do.
    return 0;
  }
  int idDest = inSer.parseInt();
  Serial.println(idDest);
  if(idDest == 0){
    // 0 Indicates Timeout on parseInt().
    Serial.println("Error: Could not parse ID.");
    flashWarning();
    delay(10);
    clearSerial(inSer);
    return 1;
  }
  int task = inSer.parseInt();
  Serial.println(task);
  if(task == blink_task){          // Blink Named MCU
    if(idDest == idNum){    // I am the MCU.  Blink.
      Serial.println("That's me!  Blink!");
      blinkID();
      clearSerial(inSer);
      return 0;
    } else {                // I am NOT the MCU. Pass it along.
      Serial.println("Not me.");
      String str = String(idDest) + " " + String(blink_task) + "\n";
      outSer.print(str);
      delay(10);
      clearSerial(inSer);
      return 0;
    }
  }
  else if (task == set_task){
    int val0 = inSer.parseInt();
    int val1 = inSer.parseInt();
    Serial.print("val0: ");
    Serial.println(val0);
    Serial.print("val1: ");
    Serial.println(val1);
    if(val0 == 0 || val1 == 0){
      // 0 Indicates Timeout on parseInt().
      Serial.println("Error: In set mode, but couldn't parseInt()");
      flashWarning();
      clearSerial(inSer);
      return 2;
    }
    if(idDest == idNum){
      targ0 = val0;
      targ1 = val1;
      Serial.println("Setting values");
      fastSet();
      clearSerial(inSer);
      return 0;
    } else {
      Serial.println("Not me, but passing on set command.");
      String str = String(idDest) + " "+String(set_task)+" " + val0 + " " + val1;
      outSer.println(str);
      clearSerial(inSer);
      return 0;
    }
  }
  else { // task not in [blink_task, set_task]
    flashWarning();
    clearSerial(inSer);
    Serial.print("Error: Unrecognized task.  Task value is: ");
    Serial.print(task);
    Serial.println(task);
    return 3;
  }
}

void setup() {
  setupPWM10();
  setupSerials();
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(serRxIn, INPUT);
  pinMode(serTxOut, OUTPUT);
}

void loop() {
  int err0 = manageInput(Serial, otherSerial);
  int err1 = manageInput(otherSerial, otherSerial);
  updateReadings();
  checkGood();
  printAllReadings();
}
