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
    neff_str = f"{S.tm_neffs[i].real:.3f}"
    Norm_Ey = np.abs(S.te_Ey[:, i] / np.max(np.abs(S.te_Ey[:, i])))
    Norm_Hx = np.abs(S.te_Hx[:, i] / np.max(np.abs(S.te_Hx[:, i])))
    Norm_Hz = np.abs(S.te_Hz[:, i] / np.max(np.abs(S.te_Hz[:, i])))

    ln = axs_TE[0].plot(S.x, Norm_Ey + i)[0]
    axs_TE[1].plot(S.x, Norm_Hx + i, color=ln.get_color())
    axs_TE[2].plot(S.x, Norm_Hz + i, color=ln.get_color())
    axs_TE[0].text(S.x.min() + 0.5, i + 0.1, neff_str, color=ln.get_color())
    axs_TE[1].text(S.x.min() + 0.5, i + 0.1, neff_str, color=ln.get_color())
    axs_TE[2].text(S.x.min() + 0.5, i + 0.1, neff_str, color=ln.get_color())

axs_TE[-1].set_xlim(S.x.min(), S.x.max())
axs_TE[-1].set_xlabel("x (µm)")
axs_TE[0].set_ylabel("$E_y$")
axs_TE[1].set_ylabel("$H_x$")
axs_TE[2].set_ylabel("$H_z$")


fig_TM, axs_TM = plt.subplots(nrows=3, sharex=True, tight_layout=True)
fig_TM.suptitle("Normalized TM Modes")
for i in range(modes):
    neff_str = f"{S.tm_neffs[i].real:.3f}"
    Norm_Hy = np.abs(S.tm_Hy[:, i] / np.max(np.abs(S.tm_Hy[:, i])))
    Norm_Ex = np.abs(S.tm_Ex[:, i] / np.max(np.abs(S.tm_Ex[:, i])))
    Norm_Ez = np.abs(S.tm_Ez[:, i] / np.max(np.abs(S.tm_Ez[:, i])))

    ln = axs_TM[0].plot(S.x, Norm_Hy + i)[0]
    axs_TM[1].plot(S.x, Norm_Ex + i, color=ln.get_color())
    axs_TM[2].plot(S.x, Norm_Ez + i, color=ln.get_color())
    axs_TM[0].text(S.x.min() + 0.5, i + 0.1, neff_str, color=ln.get_color())
    axs_TM[1].text(S.x.min() + 0.5, i + 0.1, neff_str, color=ln.get_color())
    axs_TM[2].text(S.x.min() + 0.5, i + 0.1, neff_str, color=ln.get_color())

axs_TM[-1].set_xlim(S.x.min(), S.x.max())
axs_TM[-1].set_xlabel("x (µm)")
axs_TM[0].set_ylabel("$H_y$")
axs_TM[1].set_ylabel("$E_x$")
axs_TM[2].set_ylabel("$E_z$")

## Convergence tests wrt to claddings and NRES
# NRES_values = []
# neff_convergence = []
# for NRES in np.arange(100, 500 + 50):
    # S = SlabModeSolver(λ, [n1, n2, n3], [b1, b2, b3], int(NRES))
    # S.compute_te_modes()
    # NRES_values.append(NRES)
    # neff_convergence.append(S.te_neffs[0].real)
# 
# fig, ax = plt.subplots(tight_layout=True)
# ax.plot(NRES_values, neff_convergence)
# ax.set_xlabel("NRES")
# ax.set_ylabel("Neff")
# ax.set_title("Convergence test")

plt.show()
