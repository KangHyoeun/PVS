# test_velocity_control.py
from python_vehicle_simulator.vehicles import otter
import numpy as np
import matplotlib.pyplot as plt

# Create Otter with velocity control
vehicle = otter('velocityControl', r=1.5)

# Simulation
dt = 0.02
T_sim = 50
N = int(T_sim / dt)

# Storage
time = np.zeros(N)
u_history = np.zeros(N)
r_history = np.zeros(N)

# Initial conditions
eta = np.zeros(6)
nu = np.zeros(6)
u_actual = np.zeros(2)

# References
u_ref = 1.5  # m/s
r_ref = 0.0  # rad/s

for i in range(N):
    time[i] = i * dt
    
    # Step change at t=20s
    if time[i] > 20:
        r_ref = 0.1  # ~5.7 deg/s turn
    
    # Control
    u_control = vehicle.velocityControl(nu, u_ref, r_ref, dt)
    
    # Dynamics
    [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, dt)
    
    # Store
    u_history[i] = nu[0]
    r_history[i] = nu[5] * 180/np.pi

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(time, u_history, 'b-', linewidth=2)
plt.axhline(y=u_ref, color='r', linestyle='--', label='Reference')
plt.xlabel('Time (s)')
plt.ylabel('Surge velocity (m/s)')
plt.title('Surge Velocity Control')
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(time, r_history, 'b-', linewidth=2)
plt.axhline(y=r_ref*180/np.pi, color='r', linestyle='--', label='Reference')
plt.xlabel('Time (s)')
plt.ylabel('Yaw rate (deg/s)')
plt.title('Yaw Rate Control')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()