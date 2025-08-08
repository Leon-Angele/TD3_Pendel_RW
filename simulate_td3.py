# visualizer_3d_state.py

import torch
import numpy as np
import pygame
from Td3Agent import Actor # Stelle sicher, dass Td3Agent.py im selben Ordner ist

# --- Physik-Engine (unverändert) ---
# Die Simulation selbst arbeitet weiterhin mit dem realen Winkel theta.
GRAVITY = 9.8
MASS = 1.0
LENGTH = 1.0
TIME_STEP = 0.02
ACTION_DIM = 1
MAX_ACTION = 20
M_PI = np.pi
NOISE_PERCENT = 0.0

MAX_STEPS = 250  # Maximale Anzahl an Simulationsschritten

def advance_state(state, torque):
    theta, theta_dot = state
    theta_double_dot = (GRAVITY / LENGTH) * np.sin(theta) + (1.0 / (MASS * LENGTH * LENGTH)) * torque
    theta_dot += TIME_STEP * theta_double_dot
    theta += TIME_STEP * theta_dot
    # Normalisierung des Winkels auf [-pi, pi]
    while theta > M_PI:
        theta -= 2 * M_PI
    while theta < -M_PI:
        theta += 2 * M_PI
    return np.array([theta, theta_dot])

# --- Actor-Netzwerk laden (angepasst für 3D-State) ---

# NEU: Der Zustandsraum für den Agenten hat jetzt 3 Dimensionen
STATE_DIM = 3 

actor = Actor(STATE_DIM, ACTION_DIM, MAX_ACTION)

# WICHTIG: Lade das korrekte, auf 3D-States trainierte Modell
try:
    checkpoint = torch.load("td3_agent_3d_state.pth") 
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval() # In den Evaluationsmodus schalten
    print("Agenten-Modell (td3_agent_3d_state.pth) erfolgreich geladen.")
except FileNotFoundError:
    print("FEHLER: Modelldatei 'td3_agent_3d_state.pth' nicht gefunden. Bitte den Pfad prüfen.")
    exit()


def select_action(state_for_agent):
    """ Wählt eine Aktion basierend auf der 3D-Observation. """
    state_tensor = torch.FloatTensor(state_for_agent.reshape(1, -1))
    action = actor(state_tensor).detach().cpu().numpy().flatten()
    return float(action[0])

# --- Pygame Setup (unverändert) ---
WIDTH, HEIGHT = 400, 400
CENTER = (WIDTH // 2, HEIGHT // 2)
L = 150 # Länge des Pendels in Pixel

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TD3 Pendulum (3D State)")
clock = pygame.time.Clock()

def draw_pendulum(theta):
    """ Zeichnet das Pendel basierend auf dem realen Winkel theta. """
    # Umrechnung von mathematischen Koordinaten in Pygame-Koordinaten
    x = CENTER[0] + L * np.sin(theta)
    y = CENTER[1] - L * np.cos(theta) # Minus, da y in Pygame nach unten wächst
    
    screen.fill((255, 255, 255)) # Weißer Hintergrund
    pygame.draw.line(screen, (0, 0, 0), CENTER, (int(x), int(y)), 6) # Pendelstange
    pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 20) # Pendelmasse
    pygame.display.flip()

# --- Haupt-Simulations-Loop (angepasst) ---
if __name__ == "__main__":
    physical_state = np.array([np.pi, 0.0]) # Start unten
    running = True
    step_count = 0

    while running and step_count < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        theta, theta_dot = physical_state
        agent_observation = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        
        action = select_action(agent_observation)
        noise = np.random.uniform(-NOISE_PERCENT, NOISE_PERCENT) * abs(action)
        action_noisy = action + noise

        physical_state = advance_state(physical_state, action_noisy)
        draw_pendulum(physical_state[0])
        
        clock.tick(1 / TIME_STEP)
        step_count += 1


    pygame.quit()