/*
   This is the "firmware" for a SP5T RF switch.
   
   This code is loaded onto a Metromini MCU.  The MCU is interfaced with a 
   HMC253AQS24 evaluation board using a simple breadboard.
   
   The HMC253AQS24 is a actually a SP8T RF Switch.  The output is selected by
   using three digital inputs (A, B, C) which specify the output port in binary.
   
*/

// Connections between Switch and MCU DO Pins
#define APin 3
#define BPin 4
#define CPin 5


/* In order to avoid unsightly RF cable crossings, we tweak the numbering
   slightly.  The maps below correspond to channels 1, 2, 3, 4, 5.  This
   numbering is slightly different for the input and output switches.
*/
// For Input use the line below.
// int portMap[5]  = {6, 5, 4, 3, 2};
// For Output
// int portMap[5] = {3, 4, 5, 6, 7};
int portMap[8] = {1, 2, 3, 4, 5, 6, 7, 8};

void setOutputs(int meshPort){
  // Sets the switch port.  `meshPort` expected to be in the range [1-5].
  if(meshPort<1 || meshPort>5){
    Serial.print("Meshport ");
    Serial.print(meshPort);
    Serial.println(" is not in the range [1,5].");
    Serial.println();
    return;
  }
  int switchPort = portMap[meshPort - 1];
  uint8_t n = switchPort - 1;
  uint8_t a = n%2;
  n = n/2;
  uint8_t b = n%2;
  n = n/2;
  uint8_t c = n%2;
  digitalWrite(APin, a);
  digitalWrite(BPin, b);
  digitalWrite(CPin, c);
  Serial.print("MeshPort: ");
  Serial.println(meshPort);
  Serial.print("SwitchPort: ");
  Serial.println(switchPort);
  Serial.print("binary (CBA): ");
  Serial.print(c);
  Serial.print(b);
  Serial.println(a);
  Serial.println();
}

void clearSerial(Stream &ser){
  // Cleans up the input serial buffer.
  delay(10);
  while(ser.available()){
    ser.read();
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(APin, OUTPUT);
  pinMode(BPin, OUTPUT);
  pinMode(CPin, OUTPUT);
  Serial.begin(9600);                       // Communication with Computer
  Serial.println("5Port Switch V0");      // say hello
}

void loop() {
  if(Serial.available() == 0){ // Nothing to do.
    return;
  } else {
    int meshPort = Serial.parseInt();
    setOutputs(meshPort);
    clearSerial(Serial);
  }
}
