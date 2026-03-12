import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from dm_control import mjcf
from dm_control import mujoco as dm_mujoco
import pinocchio as pin

BASE = os.path.dirname(os.path.abspath(__file__))
DT, T, NQ = 0.002, 10.0, 2
STATE_MAX = 1e6

# reference value: can be constant or time-varying
Q_REF_STATIC = np.deg2rad([-100.0, 100.0])

def q_ref(t):
    """Return reference joint positions at time t (rad). Customize as needed."""
    # Example: constant
    # return Q_REF_STATIC.copy()
    # Example: sinusoidal sweep
    omega = 0.5 * np.pi / T
    return np.deg2rad([-100.0 + 80 * np.sin(omega * t), 100.0 - 80 * np.sin(omega * t)])

# Hardcoded
Q0 = np.deg2rad([30.0, -20.0])
# TAU = np.array([0.5, 0.2])       # torque input
DAMPING, STIFFNESS = 1, 1       # damping and stiffness coefficients

# MuJoCo
root = mjcf.from_path(os.path.join(BASE, "dummy.xml"))
for j in root.find_all("joint"):
    root.actuator.add("general", name=j.name, joint=j)
physics = dm_mujoco.Physics.from_xml_string(root.to_xml_string(), assets=root.get_assets())
m, d = physics.model.ptr, physics.data.ptr
m.dof_damping[:] = DAMPING      # add damping to the MuJoCo model
m.jnt_stiffness[:] = STIFFNESS   # add stiffness to the MuJoCo model (equilibrium point = 0 -> default)
mj_dt = m.opt.timestep
n_mj = int(T / mj_dt)
t_mj = np.linspace(0, T, n_mj + 1)
q_mj = np.zeros((n_mj + 1, NQ))
d.qpos[:NQ] = q_mj[0] = Q0



for i in range(1, n_mj + 1):
    # d.ctrl[:NQ] = STIFFNESS * (q_ref - d.qpos[:NQ])       # torque input
    m.qpos_spring[:NQ] = q_ref(t_mj[i - 1])
    mujoco.mj_step(m, d)
    q_mj[i] = d.qpos[:NQ].copy()
    if np.max(np.abs(d.qpos[:NQ])) > STATE_MAX or np.max(np.abs(d.qvel[:NQ])) > STATE_MAX:
        t_mj, q_mj = t_mj[: i + 1], q_mj[: i + 1]
        print(f"MuJoCo stopped at t={t_mj[-1]:.4f}s")
        break


# Pinocchio
pin_model = pin.buildModelFromUrdf(os.path.join(BASE, "dummy.urdf"))
pin_data = pin_model.createData()

def dynamics(q, v, tau, q_ref_t):
    q, v, tau = np.asarray(q).reshape(-1), np.asarray(v).reshape(-1), np.asarray(tau).reshape(-1)
    M = pin.crba(pin_model, pin_data, q)
    b = pin.nonLinearEffects(pin_model, pin_data, q, v)
    vdot = np.linalg.solve(M, tau - b - DAMPING * v - STIFFNESS * (q - q_ref_t))   # (M + B)*vdot + b + damping*v + stiffness*(q - q_eq) = tau
    return v, vdot

def rk4(q, v, tau, dt, q_ref_t):
    k1q, k1v = dynamics(q, v, tau, q_ref_t)
    k2q, k2v = dynamics(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v, tau, q_ref_t)
    k3q, k3v = dynamics(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v, tau, q_ref_t)
    k4q, k4v = dynamics(q + dt * k3q, v + dt * k3v, tau, q_ref_t)
    return q + (dt / 6) * (k1q + 2 * k2q + 2 * k3q + k4q), v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)

n_pin = int(T / DT)
t_pin = np.linspace(0, T, n_pin + 1)
q_pin = np.zeros((n_pin + 1, NQ))
q_pin[0] = Q0
q, v = Q0.copy(), np.zeros(NQ)
for k in range(n_pin):
    q, v = rk4(q, v, np.zeros(NQ), DT, q_ref(t_pin[k]))       # RK4 solver, without torque input
    q_pin[k + 1] = q.copy()
    if np.max(np.abs(q)) > STATE_MAX or np.max(np.abs(v)) > STATE_MAX:
        t_pin, q_pin = t_pin[: k + 2], q_pin[: k + 2]
        print(f"Pinocchio stopped at t={t_pin[-1]:.4f}s")
        break

# Compare & plot
q_mj_ip = np.column_stack([np.interp(t_pin, t_mj, q_mj[:, j]) for j in range(NQ)])
err = np.abs(q_mj_ip - q_pin)
print(f"Max |delta_q|: {np.max(err):.6f}, mean: {np.mean(err):.6f}")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
for j in range(NQ):
    ax1.plot(t_pin, q_pin[:, j], label=f"Pin q{j+1}", alpha=0.8)
    ax1.plot(t_pin, q_mj_ip[:, j], "--", label=f"MJ q{j+1}", alpha=0.8)
    ax2.plot(t_pin, err[:, j], label=f"|delta_q{j+1}|")
ax1.set_ylabel("q (rad)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.set_ylabel("error (rad)")
ax2.set_xlabel("time (s)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
