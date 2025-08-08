
import serial
import time
import json
import numpy as np

# --- Serielle Konfiguration ---
ARDUINO_PORT = 'COM7'  # Passe diesen Port an!
BAUDRATE = 921600


class env:

    def __init__(self):
        try:
            print(f"Verbinde mit Arduino auf Port {ARDUINO_PORT}...")
            # Timeout von 2s, falls der Arduino mal nicht antwortet
            self.ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=2)
            time.sleep(0.5)  # Kurze Pause, damit der Arduino sich initialisieren kann
            print("Verbindung erfolgreich.")

        except KeyboardInterrupt:
            print("\nProgramm durch Benutzer beendet.")
            self.close();
        except serial.SerialException as e:
            print(f"Fehler: Konnte den Port {ARDUINO_PORT} nicht öffnen. {e}")
            self.close();
            

    def send_action(self, action):
        #print(f"Sende Aktion: {action}")
        tx_action(self.ser, action)

    def get_state(self):
        state = rx_state(self.ser)
        #print(f"Empfangener Zustand: {state}")
        return state.get('step'), np.array([state.get('cos_theta'), state.get('sin_theta'), state.get('angular_velocity')], dtype=np.float32), state.get('angle_for_reward'), state.get('done')



    def close(self):
         print("\nProgramm beendet.")
         if 'ser' in locals() and self.ser.is_open:
                action = {"speed": 0}
                tx_action(self.ser, action)
                self.ser.close()
                print("Verbindung sicher geschlossen.")


    def start_episode(self):
 
#        1. Startbefehl senden
        #print("Sende 'S', um die Episode zu starten...")
        self.ser.write(b'S')
        self.ser.flush() # Sicherstellen, dass der Befehl sofort gesendet wird
        
        time.sleep(1)  # Kurze Pause, damit der Arduino den Reset verarbeitet
        # 2. Auf den initialen Zustand nach dem Reset warten
        initial_state = json.loads(self.ser.readline().decode('utf-8').strip())
        #print(f"Initialen Zustand empfangen: {initial_state}")




    #state_doc["step"] = state.step;
    #state_doc["cos_theta"] = cos(state.angle);
    #state_doc["sin_theta"] = sin(state.angle);
    #state_doc["angular_velocity"] = state.angular_velocity;
    #state_doc["angle_for_reward"] = state.angle;
    #state_doc["done"] = done;
    #state_doc["over_time"] = state.overTime;
    


def tx_action(ser_instance, action_dict):
    """
    Sendet eine Aktion an den Arduino.
    """

    #action_dict = 0
    try:
        # Aktion als JSON senden
        ser_instance.write(json.dumps({"speed": action_dict}).encode('utf-8') + b'\n')
        #ser_instance.flush()  # Sicherstellen, dass der Befehl sofort gesendet wird
    except serial.SerialException as e:
        print(f"Fehler beim Senden der Aktion: {e}")

def rx_state(ser_instance):
    """
    Empfängt und parst den Zustand vom Arduino.
    """
    try:
        # Auf Antwort warten (blockiert, bis eine Zeile empfangen wird)
        response_json = ser_instance.readline().decode('utf-8').strip()
        
        if response_json:
            return json.loads(response_json)
        else:
            print("WARNUNG: Leere Antwort vom Arduino erhalten.")
            return None
    except (serial.SerialException, json.JSONDecodeError) as e:
        print(f"Fehler beim Empfangen des Zustands: {e}")
        return None
    

            