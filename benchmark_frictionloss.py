import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco

BASE = os.path.dirname(os.path.abspath(__file__))
coulomb = 0.5

# --- Model WITH friction ---
m_fric = mujoco.MjModel.from_xml_path(os.path.join(BASE, "prismatic.xml"))
d_fric = mujoco.MjData(m_fric)
m_fric.dof_frictionloss[:] = coulomb

d_fric.qpos[:] = 0.0
d_fric.qvel[:] = 0.5

n = int(10.0 / 0.002)
t = np.linspace(0, 10.0, n + 1)
q_fric = np.zeros((n + 1, 1))
v_fric = np.zeros((n + 1, 1))
q_fric[0] = d_fric.qpos.copy()
v_fric[0] = d_fric.qvel.copy()

for i in range(1, n + 1):
    d_fric.ctrl[:] = 1
    if i > 3/0.002:
        d_fric.ctrl[:] = 0.0
    mujoco.mj_step(m_fric, d_fric)
    q_fric[i] = d_fric.qpos.copy()
    v_fric[i] = d_fric.qvel.copy()

# --- Model WITHOUT friction ---
m_nofric = mujoco.MjModel.from_xml_path(os.path.join(BASE, "prismatic.xml"))
d_nofric = mujoco.MjData(m_nofric)
m_nofric.dof_frictionloss[:] = 0.0

d_nofric.qpos[:] = 0.0
d_nofric.qvel[:] = 0.5

q_nofric = np.zeros((n + 1, 1))
v_nofric = np.zeros((n + 1, 1))
q_nofric[0] = d_nofric.qpos.copy()
v_nofric[0] = d_nofric.qvel.copy()

for i in range(1, n + 1):
    action_force = 1
    if i > 3/0.002:
        action_force = 0.0

    d_nofric.ctrl[:] = action_force - coulomb*np.sign(d_nofric.qvel[0])
    


    mujoco.mj_step(m_nofric, d_nofric)
    q_nofric[i] = d_nofric.qpos.copy()
    v_nofric[i] = d_nofric.qvel.copy()

# --- Plot ---
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
axes[0].plot(t, q_fric[:, 0], label="With friction")
axes[0].plot(t, q_nofric[:, 0], "--", label="Without friction")
axes[0].set_ylabel("Position (m)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, v_fric[:, 0], label="With friction")
axes[1].plot(t, v_nofric[:, 0], "--", label="Without friction")
axes[1].set_ylabel("Velocity (m/s)")
axes[1].set_xlabel("Time (s)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
