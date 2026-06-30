#include <SoftwareSerial.h>

SoftwareSerial BT(2, 3);

int in1 = 6;
int in2 = 7;
int in3 = 8;
int in4 = 9;
int pwmd = 10;
int pwme = 11;

char msg;

void setup() {
  BT.begin(9600);
  Serial.begin(9600);

  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(pwmd, OUTPUT);
  pinMode(pwme, OUTPUT);
}

void loop() {
  if (BT.available()) {
    msg = BT.read();
    Serial.println(msg);

    if (msg == 'L') {
      analogWrite(pwmd, 255);
      analogWrite(pwme, 255);

      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
    }

    if (msg == 'R') {
      analogWrite(pwmd, 255);
      analogWrite(pwme, 255);

      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
    }

    if (msg == 'B') {
      analogWrite(pwmd, 255);
      analogWrite(pwme, 255);

      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
    }
    if (msg == 'I') {
      analogWrite(pwmd, 150);
      analogWrite(pwme, 255);

      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
    }
    if (msg == 'J') {
      analogWrite(pwmd, 255);
      analogWrite(pwme, 120);

      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
    }

    if (msg == 'F') {
      analogWrite(pwmd, 255);
      analogWrite(pwme, 255);

      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
    }

  
    if (msg == 'G') {
      analogWrite(pwmd, 150);
      analogWrite(pwme, 255);

      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
    }


    if (msg == 'H') {
      analogWrite(pwmd, 255);
      analogWrite(pwme, 120);

      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
    }

    if (msg == 'S') {
      analogWrite(pwmd, 0);
      analogWrite(pwme, 0);

      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
    }
  }
}