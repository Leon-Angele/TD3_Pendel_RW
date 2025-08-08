#include <ArduinoJson.h>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

// ---------------------------
// 1. Datenstrukturen
// ---------------------------
struct state_t {
    float angle;
    float angular_velocity;
};

struct action_t {
    float torque;
};

// ---------------------------
// 2. Simulations-Logik (unverändert)
// ---------------------------
state_t current_state = {M_PI, 0.0};
const float GRAVITY = 9.8;
const float MASS = 1.0;
const float LENGTH = 1.0;
const float TIME_STEP = 0.02;

void reset_environment() {
    current_state.angle = M_PI;
    current_state.angular_velocity = 0.0;
}

void advance_state(action_t action) {
    float torque = action.torque;
    float theta = current_state.angle;
    float theta_dot = current_state.angular_velocity;

    float theta_double_dot = (GRAVITY / LENGTH) * sin(theta) + (1.0 / (MASS * LENGTH * LENGTH)) * torque;
    
    theta_dot += TIME_STEP * theta_double_dot;
    theta += TIME_STEP * theta_dot;

    // --- NEUE, ROBUSTERE NORMALISIERUNG ---
    // Diese Methode vermeidet den Fehler bei exakt PI.
    theta = fmod(theta + M_PI, 2 * M_PI);
    if (theta < 0) {
        theta += 2 * M_PI;
    }
    theta = theta - M_PI;
    // --- ENDE DER NEUEN NORMALISIERUNG ---

    current_state.angle = theta;
    current_state.angular_velocity = theta_dot;
}

// NEU: Angepasste Funktion zum Senden des Zustands, jetzt mit Aktion
void send_state_and_done(bool done, action_t action) {
    // Die Dokumentgröße leicht erhöht, um sicher Platz für das neue Feld zu haben
    StaticJsonDocument<300> doc;

    doc["cos_theta"] = cos(current_state.angle);
    doc["sin_theta"] = sin(current_state.angle);
    doc["angular_velocity"] = current_state.angular_velocity;
    
    // Wichtig: Sende den rohen Winkel separat für die Belohnungsberechnung in Python
    doc["angle_for_reward"] = current_state.angle;
    
    // NEU: Füge die Aktion hinzu, die zu diesem Zustand geführt hat
    doc["action_torque"] = action.torque;
    
    doc["done"] = done;
    
    serializeJson(doc, Serial);
    Serial.println(); // Wichtig: Newline als Endzeichen
}


// Setup bleibt exakt gleich
void setup() {
    Serial.begin(921600);
    while (!Serial) { ; }
    reset_environment();
}

// DIES IST DIE KORREKTE VERSION
void loop() {
    if (Serial.available() > 0 && Serial.read() == 'S') {
        reset_environment();
        int step_count = 0;
        bool done = false;

        // WICHTIG: Kein initialer Zustand senden!

        while (!done) {
            // 1. Warte auf Aktion vom PC
            while (Serial.available() == 0) { }
            String incoming_string = Serial.readStringUntil('\n');
            StaticJsonDocument<200> action_doc;
            deserializeJson(action_doc, incoming_string);
            action_t received_action = { action_doc["torque"] };

            // 2. Wende Aktion an
            advance_state(received_action);
            step_count++;

            // 3. Prüfe, ob die Episode beendet ist
            if (step_count >= 250) {
                done = true;
            }

            // 4. Sende neuen Zustand und die zuletzt ausgeführte Aktion
            send_state_and_done(done, received_action);
        }
    }
}