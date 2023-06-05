import numpy as np
import matplotlib.pyplot as plt
from fdfd_mode_solver.refractive_indices import *


x = 0
T = 25
λ = np.linspace(0.5e-6, 2e-6, 10001)
y1 = nGaAs_Skauli(x, λ, Temperature=T)
y2 = nAlGaAs_Gehrsitz(x, λ, Temperature=T)
y3 = nAlGaAs_Afromowitz(x, λ)
y4 = nAlGaAs_Adachi(x, λ)
y5 = nAlGaAs_Kim(x, λ, Temperature=T)
y6 = nSiO2_Malitson(λ)
y7 = nSiO2_Arosa(λ)
y8 = nSi3N4_Luke(λ)
y9 = n_ordinary_LiNbO3_Zelmon(λ)
y10 = n_extraordinary_LiNbO3_Zelmon(λ)

# GaAs
fig, ax = plt.subplots()
ax.plot(λ * 1e6, y1.real, label="Skauli (2003)")
ax.plot(λ * 1e6, y2.real, label="Gehrsitz (2000)")
ax.plot(λ * 1e6, y3.real, label="Afromowitz (1974)")
ax.plot(λ * 1e6, y4.real, label="Adachi (1985)")
ax.plot(λ * 1e6, y5.real, label="Kim (2007)")
ax.legend(frameon=False)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("GaAs refractive index")

# SiO2
fig, ax = plt.subplots()
ax.plot(λ * 1e6, y6.real, label="Malitson (1965)")
ax.plot(λ * 1e6, y7.real, label="Arosa (2020)")
ax.legend(frameon=False)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("SiO$_2$ refractive index")

# Si3N4
fig, ax = plt.subplots()
ax.plot(λ * 1e6, y8.real, label="Luke (2015)")
ax.legend(frameon=False)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Si$_3$N$_4$ refractive index")

# LiNbO3
fig, ax = plt.subplots()
ax.plot(λ * 1e6, y9.real, label="n$_\\mathrm{o}$ Zelmon (1997)")
ax.plot(λ * 1e6, y10.real, label="n$_\\mathrm{e}$ Zelmon (1997)")
ax.legend(frameon=False)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("LiNbO$_3$ refractive index")


# Al(x)Ga(1-x)As
fig.tight_layout()

λ = np.linspace(0.95e-6, 2.6e-6, 10001)
y = nAlGaAs_Gehrsitz(0, λ, Temperature=20)
z = nAlGaAs_Gehrsitz(0.8, λ, Temperature=20)

fig, ax = plt.subplots()
ax.plot(λ * 1e6, y.real, label="GaAs")
ax.plot(λ * 1e6, z.real, label="Al$_{0.8}$Ga$_{0.2}$As")

ax.legend()
ax.set_xlim(0.95, 2.4)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Real refractive index at $T=20$°")
fig.tight_layout()

# GaP
λ = np.linspace(0.7e-6, 12.5e-6, 1001)
Temperatures = [78, 150, 200, 250, 300, 350, 400, 450]  # [K]

fig, ax = plt.subplots(tight_layout=True)
for T in Temperatures:
    n = nGaP_Wei(λ, T)
    ax.plot(λ * 1e6, n.real, label=f"T = {T}K")
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("GaP refractive index")
ax.legend()
ax.set_xlim(0.7, 12.5)
ax.set_ylim(2.90, 3.25)

plt.show()