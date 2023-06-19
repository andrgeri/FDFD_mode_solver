import numpy as np
import matplotlib.pyplot as plt
from fdfd_mode_solver import SlabModeSolver

# Material parameters
λ = 1.55  # [µm], free space wavelength
n1 = 1.445  # [-], top cladding refractive index
n2 = 3.476  # [-], core refractive index
n3 = 1.445  # [-], bottom cladding refractive index

# Thickness of each layer
b1 = 5 * λ  # top cladding thickness
b2 = 0.22  # [µm], core thickness
b3 = 5 * λ  # bottom cladding thickness

# Grid
NRES = 500  # grid resolution
dx = λ / NRES  # grid spacing

# Number of modes to calculate
modes = 2

# Eigenmodes
S = SlabModeSolver(λ, [n1, n2, n3], [b1, b2, b3], NRES)
S.compute_te_modes(nmodes=modes)
S.compute_tm_modes(nmodes=modes)

# Visualize modes
fig_TE, axs_TE = plt.subplots(nrows=3, sharex=True, tight_layout=True)
fig_TE.suptitle("Normalized TE Modes")
for i in range(modes):
    TE_mode = S.TE_modes[i]
    neff_str = f"{TE_mode.neff.real:.3f}"
    Norm_Ey = np.abs(TE_mode.Ey / np.max(np.abs(TE_mode.Ey)))
    Norm_Hx = np.abs(TE_mode.Hx / np.max(np.abs(TE_mode.Hx)))
    Norm_Hz = np.abs(TE_mode.Hz / np.max(np.abs(TE_mode.Hz)))

    ln = axs_TE[0].plot(TE_mode.x, Norm_Ey + i)[0]
    color = ln.get_color()
    axs_TE[1].plot(TE_mode.x, Norm_Hx + i, color=color)
    axs_TE[2].plot(TE_mode.x, Norm_Hz + i, color=color)
    axs_TE[0].text(TE_mode.x.min() + 0.5, i + 0.1, neff_str, color=color)
    axs_TE[1].text(TE_mode.x.min() + 0.5, i + 0.1, neff_str, color=color)
    axs_TE[2].text(TE_mode.x.min() + 0.5, i + 0.1, neff_str, color=color)

axs_TE[-1].set_xlim(TE_mode.x.min(), TE_mode.x.max())
axs_TE[-1].set_xlabel("x (µm)")
axs_TE[0].set_ylabel("$E_y$")
axs_TE[1].set_ylabel("$H_x$")
axs_TE[2].set_ylabel("$H_z$")

TE0 = S.TE_modes[0]
fig, axs = plt.subplots(ncols=3)
TE0.plot_Ey(axs[0])
TE0.plot_Hx(axs[1])
TE0.plot_Hz(axs[2])


fig_TM, axs_TM = plt.subplots(nrows=3, sharex=True, tight_layout=True)
fig_TM.suptitle("Normalized TM Modes")
for i in range(modes):
    TM_mode = S.TM_modes[i]
    neff_str = f"{TM_mode.neff.real:.3f}"
    Norm_Hy = np.abs(TM_mode.Hy / np.max(np.abs(TM_mode.Hy)))
    Norm_Ex = np.abs(TM_mode.Ex / np.max(np.abs(TM_mode.Ex)))
    Norm_Ez = np.abs(TM_mode.Ez / np.max(np.abs(TM_mode.Ez)))

    ln = axs_TM[0].plot(TM_mode.x, Norm_Hy + i)[0]
    color = ln.get_color()
    axs_TM[1].plot(TM_mode.x, Norm_Ex + i, color=color)
    axs_TM[2].plot(TM_mode.x, Norm_Ez + i, color=color)
    axs_TM[0].text(TM_mode.x.min() + 0.5, i + 0.1, neff_str, color=color)
    axs_TM[1].text(TM_mode.x.min() + 0.5, i + 0.1, neff_str, color=color)
    axs_TM[2].text(TM_mode.x.min() + 0.5, i + 0.1, neff_str, color=color)

axs_TM[-1].set_xlim(TM_mode.x.min(), TM_mode.x.max())
axs_TM[-1].set_xlabel("x (µm)")
axs_TM[0].set_ylabel("$H_y$")
axs_TM[1].set_ylabel("$E_x$")
axs_TM[2].set_ylabel("$E_z$")

TM0 = S.TM_modes[0]
fig, axs = plt.subplots(ncols=3)
TM0.plot_Hy(axs[0])
TM0.plot_Ex(axs[1])
TM0.plot_Ez(axs[2])

# Convergence tests wrt to claddings and NRES
NRES_values = []
neff_convergence = []
for NRES in np.arange(100, 500 + 50, 50):
    S = SlabModeSolver(λ, [n1, n2, n3], [b1, b2, b3], int(NRES))
    S.compute_te_modes()
    NRES_values.append(NRES)
    neff_convergence.append(S.TE_modes[0].neff.real)

fig, ax = plt.subplots(tight_layout=True)
ax.plot(NRES_values, neff_convergence)
ax.set_xlabel("NRES")
ax.set_ylabel("Neff")
ax.set_title("Convergence test")

plt.show()
