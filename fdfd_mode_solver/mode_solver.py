import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigs, inv
from scipy.constants import c, epsilon_0 as ε0, mu_0 as µ0


class AbstractSlabMode():
    def __init__(self, λ: float, x, εr, µr):
        self.λ = λ
        self.x = x
        self.εr = εr
        self.µr = µr

    def plot_epsilon(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ax.plot(self.x, self.εr)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_xlabel("x")
        ax.set_ylabel("Relative permittivity ε$_r$")

    def plot_n(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ax.plot(self.x, np.sqrt(self.εr))
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_xlabel("x")
        ax.set_ylabel("Refractive index $n$")

    def plot_mu(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ax.plot(self.x, self.µr)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_xlabel("x")
        ax.set_ylabel("Relative permeability µ$_r$")


class TESlabMode(AbstractSlabMode):
    def __init__(self, λ: float, x, εr, µr, neff, Ey, Hx, Hz):
        super().__init__(λ, x, εr, µr)
        self.neff = neff
        self.Ey = Ey
        self.Hx = Hx
        self.Hz = Hz

    def plot_Ey(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ϕ = np.abs(self.Ey) / np.max(np.abs(self.Ey))
        s = f"{self.neff.real:.4f} + j{self.neff.imag:.4f}"

        ax.plot(self.x, ϕ)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_title(r"$n_\mathrm{eff}$ = " + s)
        ax.set_xlabel("x")
        ax.set_ylabel("Normalized $E_y$")

    def plot_Hx(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ϕ = np.abs(self.Hx) / np.max(np.abs(self.Hx))
        s = f"{self.neff.real:.4f} + j{self.neff.imag:.4f}"

        ax.plot(self.x, ϕ)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_title(r"$n_\mathrm{eff}$ = " + s)
        ax.set_xlabel("x")
        ax.set_ylabel("Normalized $H_x$")

    def plot_Hz(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ϕ = np.abs(self.Hz) / np.max(np.abs(self.Hz))
        s = f"{self.neff.real:.4f} + j{self.neff.imag:.4f}"

        ax.plot(self.x, ϕ)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_title(r"$n_\mathrm{eff}$ = " + s)
        ax.set_xlabel("x")
        ax.set_ylabel("Normalized $H_z$")


class TMSlabMode(AbstractSlabMode):
    def __init__(self, λ: float, x, εr, µr, neff, Hy, Ex, Ez):
        super().__init__(λ, x, εr, µr)
        self.neff = neff
        self.Hy = Hy
        self.Ex = Ex
        self.Ez = Ez

    def plot_Hy(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ϕ = np.abs(self.Hy) / np.max(np.abs(self.Hy))
        s = f"{self.neff.real:.4f} + j{self.neff.imag:.4f}"

        ax.plot(self.x, ϕ)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_title(r"$n_\mathrm{eff}$ = " + s)
        ax.set_xlabel("x")
        ax.set_ylabel("Normalized $H_y$")

    def plot_Ex(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ϕ = np.abs(self.Ex) / np.max(np.abs(self.Ex))
        s = f"{self.neff.real:.4f} + j{self.neff.imag:.4f}"

        ax.plot(self.x, ϕ)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_title(r"$n_\mathrm{eff}$ = " + s)
        ax.set_xlabel("x")
        ax.set_ylabel("Normalized $E_x$")

    def plot_Ez(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True)

        ϕ = np.abs(self.Ez) / np.max(np.abs(self.Ez))
        s = f"{self.neff.real:.4f} + j{self.neff.imag:.4f}"

        ax.plot(self.x, ϕ)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_title(r"$n_\mathrm{eff}$ = " + s)
        ax.set_xlabel("x")
        ax.set_ylabel("Normalized $E_z$")


class SlabModeSolver():

    def __init__(self, λ: float, ns: list, ts: list, NRES: int):
        self.λ = λ
        self.ns = ns
        self.ts = ts
        self.NRES = NRES

        if λ <= 0:
            raise ValueError("The wavelength must be positive")
        if len(ns) != len(ts):
            raise ValueError("The arrays ns and ts must have the same length")
        if len(ns) < 3:
            raise ValueError("The arrays must have at least three elements")
        if any(t < 0 for t in ts):
            raise ValueError("Each element of ts must be positive")
        if not isinstance(NRES, int):
            raise TypeError("NRES must be an integer")

    def compute_grid(self):
        # Compute x grid
        self.dx = self.λ / self.NRES
        Sx = sum(self.ts)
        self.Nx = int(np.ceil(Sx / self.dx))
        Sx = self.Nx * self.dx
        x = np.arange(0.5, self.Nx - 0.5 + 1, 1) * self.dx
        self.x = x - np.mean(x)

        # Build refractive index profile N
        self.N = np.empty(self.Nx)
        tmp = x.min()
        for t_, n_ in zip(self.ts, self.ns):
            idx = np.where((x >= tmp) & (x <= tmp + t_))
            self.N[idx] = n_
            tmp += t_
        # self.N = np.flip(self.N)  # NOT NEEDED

        return self.x, self.N

    def compute_te_modes(self, nmodes=1):
        if not isinstance(nmodes, int) or nmodes <= 0:
            raise TypeError("The number of modes must be a positive integer")
        self.nmodes = nmodes

        self.compute_grid()

        # Calculate k0
        self.k0 = 2 * np.pi / self.λ

        # Build derivative matrix on Yee's grid
        offsets = [0, 1]
        DEX = diags([-1, 1], offsets=offsets, shape=(self.Nx, self.Nx))
        DEX /= (self.k0 * self.dx)
        DHX = (- DEX.T)

        # Magnetic permeability
        muX = eye(self.Nx).tocsc()
        muY = muX
        muZ = muX

        # Make N diagonal
        N_ = diags(self.N, offsets=0, shape=(self.Nx, self.Nx))
        # This is valid only because N is a diagonal matrix!
        N2 = (N_**2).tocsc()
        muZ_inv = diags(1 / muZ.diagonal())

        # A = -µX ⋅ DhX / µZ ⋅ DeX + ε
        A = - muX @ (DHX @ muZ_inv @ DEX + N2)
        # A = - muX.dot(DHX.dot(muZ_inv).dot(DEX) + N2)

        eigvals, eigvecs = eigs(A, k=self.nmodes, sigma=-max(self.ns)**2)
        NEFF = -1j * np.sqrt(eigvals)
        # Sort modes wrt their effective index
        ind = np.argsort(np.real(NEFF))
        MODE = eigvecs[:, ind]
        NEFF = NEFF[ind]

        self.te_neffs = NEFF
        self.te_Ey = MODE
        self.te_Hx = np.empty(self.te_Ey.shape, dtype=complex)
        self.te_Hz = np.empty(self.te_Ey.shape, dtype=complex)
        for i in range(self.nmodes):
            # self.te_Hx[:, i] = -self.te_neffs[i] * np.sqrt(ε0 / µ0) * inv(N2) @ self.te_Ey[:, i]
            # self.te_Hz[:, i] = 1j * self.λ / (2 * np.pi * c / µ0) * inv(muX) * np.gradient(self.te_Ey[:, i], self.x)
            self.te_Hx[:, i] = inv(N2) @ self.te_Ey[:, i]
            self.te_Hz[:, i] = inv(muX) * np.gradient(self.te_Ey[:, i], self.x)

        return self.te_neffs, self.te_Ey, self.te_Hx, self.te_Hz

    def compute_tm_modes(self, nmodes=1):
        if not isinstance(nmodes, int) or nmodes <= 0:
            raise TypeError("The number of modes must be a positive integer")
        self.nmodes = nmodes

        self.compute_grid()

        # Calculate k0
        self.k0 = 2 * np.pi / self.λ

        # Build derivative matrix on Yee's grid
        offsets = [0, 1]
        DEX = diags([-1, 1], offsets=offsets, shape=(self.Nx, self.Nx))
        DEX /= (self.k0 * self.dx)
        DHX = (- DEX.T)

        # Magnetic permeability
        muX = eye(self.Nx)
        muY = muX
        muZ = muX

        # Make N diagonal
        N_ = diags(self.N, offsets=0, shape=(self.Nx, self.Nx))
        N2 = (N_**2).tocsc()
        N2_inv = diags(1 / N2.diagonal())

        # A = -ε ⋅ DeX / ε ⋅ DhX + µY
        A = - N2 @ (DEX @ N2_inv @ DHX + muY)

        eigenvalues, eigenmodes = eigs(A, k=self.nmodes, sigma=-max(self.ns)**2)
        NEFF = -1j * np.sqrt(eigenvalues)
        # Sort modes wrt their effective index
        ind = np.argsort(np.real(NEFF))
        MODE = eigenmodes[:, ind]
        NEFF = NEFF[ind]

        # Eigenvalues in descending order
        self.tm_neffs = NEFF
        self.tm_Hy = MODE
        self.tm_Ex = np.empty(self.tm_Hy.shape, dtype=complex)
        self.tm_Ez = np.empty(self.tm_Hy.shape, dtype=complex)

        # Missing prefactors!
        for i in range(self.nmodes):
            # self.tm_Ex[:, i] = (self.tm_neffs[i] / c / ε0) * inv(N2) @ self.tm_Hy[:, i]
            # self.tm_Ez[:, i] = 1j * self.λ / (2 * np.pi * c / ε0) * inv(N2) @ np.gradient(self.tm_Hy[:, i], self.x)
            self.tm_Ex[:, i] = inv(N2) @ self.tm_Hy[:, i]
            self.tm_Ez[:, i] = inv(N2) @ np.gradient(self.tm_Hy[:, i], self.x)

        return self.tm_neffs, self.tm_Hy, self.tm_Ex, self.tm_Ez
