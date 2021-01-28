#include <FastPID.h>
//pid settings and gains
#define OUTPUT_MIN 0
#define OUTPUT_MAX 0x03ff
#define KP 0.8
#define KI 0.0
#define KD 0.025
#define HZ 100
#define BITS 10

#define readPin1    A0
#define readPin0    A1
#define writePin1   10
#define writePin0   9
#define writePin1F  6
#define writePin0F  5

int targ0, targ1, valRead0, valRead1, valOut0, valOut1, dValOut0, dValOut1, targ0New, targ1New;
long lastUpdateTime, now, nowSetting;
//FastPID myPID( KP,  KI,    KD, HZ, BITS, true);
//FastPID myPID(0.8, 0.0, 0.025, HZ, BITS, true);
FastPID myPID(2.0, 0.0, 0.040, HZ, BITS, true);

unsigned long secs, secsOld;

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

void kickPin(int pin, int v) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, v);
  pinMode(pin, INPUT);
}

void setTargetsRandom() {
  long now = millis();
  if (now - nowSetting > 1000) {
    targ0 = random(0, 0x03ff);
    nowSetting = now;
  }
}

void setTargetsStep() {
  unsigned long secs = round(millis() / 1000);
  if (secs % 2 == 0) {
    targ0 = targ1 = 0x0300;
  } else {
    targ0 = targ1 = 0x00ff;
  }
}

void updateReadings() {
  valRead0 = analogRead(A0);
  valRead1 = analogRead(A1);
}

void checkGood() {
  if (abs(valRead0 - targ0) < 2) {
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void adjust1() {
  now = micros();
  if ((now - lastUpdateTime) / 1.e6 > 1. / HZ) {
    int unConst0 = valOut0 + myPID.step(targ0, valRead0);
    valOut0 = constrain(unConst0, 0, 0x03ff);
    lastUpdateTime = now;
    analogWrite10(writePin0, valOut0);
  }

}

void adjust2() {
  analogWrite10(writePin0, targ0);
}

void adjust3() {

  if (valRead0 - targ0 > 20) {
    valOut0 = 0;
  } else if (valRead0 - targ0 < -20) {
    valOut0 = 0x03ff;
  } else {
    valOut0 = targ0;
  }
  analogWrite10(writePin0, valOut0);
}

void adjust4() {
  int error = valRead0 - targ0;
  valOut0 = constrain(targ0 - 0.005 * pow(error, 3), 0, 0x03ff);
  analogWrite10(writePin0, valOut0);
}

void adjust5() {
  pinMode(writePin0F, OUTPUT);
  analogWrite10(writePin0, targ0);
  analogWrite(writePin0F, (uint8_t)(targ0 >> 2));
  delay(20);
  pinMode(writePin0F, INPUT);
  analogWrite10(writePin0, targ0);
}

void printAllReadings() {
  Serial.print(targ0); Serial.print(", ");
  Serial.print(valRead0); Serial.print(", ");
  Serial.println(valOut0);
}

void printControl() {
  Serial.println(valRead0);
}

void setup() {
  setupPWM10();
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  targ0 = 0x03ff;    // 0x03ff is largest value
      secs = secsOld = round(millis()/1000);
      while(secsOld == secs){
        secsOld = secs;
        secs = round(millis()/1000);
      }
      if(secs%2 == 0){
        targ0New = 0x03ff;
      } else {
        targ0New = 0x01ff;
      }
      targ0 = targ0New;
  adjust5();
  printControl();
}
