import numpy as np
from scipy.constants import h, c, e as q


def coth(x):
    return np.cosh(x) / np.sinh(x)


def nGaAs_Skauli(x, λ, Temperature=22):
    """GaAs refractive index model according to [1].

    Parameters
    ----------
    x : float
        Al molar fraction (0 <= x <= 1)
    λ : float
        Wavelength in [m] (λ > 0)
    Temperature : float, optional (default to 22°C)

    Returns
    -------
    n + ik : complex
        Complex refractive index

    References
    ----------
    .. [1] T. Skauli, P. S. Kuo, K. L. Vodopyanov, T. J. Pinguet, O. Levi,
       L. A. Eyres, J. S. Harris, M. M. Fejer, B. Gerard, L. Becouarn,
       and E. Lallier, "Improved dispersion relations for GaAs and applications
       to nonlinear optics," Journal of Applied Physics 94, 6447–6455 (2003).
.
    """
    ΔT = 22 - Temperature
    ħω = h * c / q / λ  # [eV]

    A = 0.689578
    G3 = 0.00218176
    ε2 = 12.999386
    E0 = 1.425000 - 0.00037164 * ΔT - 7.497e-7 * ΔT**2  # [eV]
    E1 = 2.400356 - 0.00051458 * ΔT  # [eV]
    E2 = 7.691979 - 0.00046545 * ΔT  # [eV]
    E3 = 0.034303 + 0.00001136 * ΔT  # [eV]

    _1st = A / np.pi * np.log((E1**2 - ħω**2) / (E0**2 - ħω**2))
    _2nd = ε2 / np.pi * np.log((E2**2 - ħω**2) / (E1**2 - ħω**2))
    # _1st = A / np.pi * np.sign((E1**2 - ħω**2) / (E0**2 - ħω**2)) * np.log(np.abs((E1**2 - ħω**2) / (E0**2 - ħω**2)))
    # _2nd = ε2 / np.pi * np.sign((E2**2 - ħω**2) / (E1**2 - ħω**2)) * np.log(np.abs((E2**2 - ħω**2) / (E1**2 - ħω**2)))
    _3rd = G3 / (E3**2 - ħω**2)
    ε = 1 + _1st + _2nd + _3rd
    return np.sqrt(ε, dtype=complex)


def nAlGaAs_Gehrsitz(x, λ, Temperature=20):
    """Al(x)Ga(1-x)As refractive index model according to [1].

    Parameters
    ----------
    x : float
        Al molar fraction (0 <= x <= 1)
    λ : float
        Wavelength in [m] (λ > 0)

    Returns
    -------
    n + ik : complex
        Complex refractive index

    References
    ----------
    .. [1] S. Gehrsitz, F. K. Reinhart, C. Gourgon, N. Herres, A. Vonlanthen,
       and H. Sigg, "The refractive index of Al$_{x}$Ga$_{1-x}$As below the
       band gap: Accurate determination and empirical modeling," Journal of
       Applied Physics 87, 7825–7837 (2000).

    """
    T = Temperature + 273.15
    µm = 1.239856          # [eV µm], energy conversion to µm⁻¹
    kB = 0.0861708 * 1e-3  # [eV/K], Boltzmann constant
    V_T = 2 * kB * T       # [eV]

    E = h * c / q / λ / µm  # [µm⁻¹]

    # Energy bandgap (in Γ) for GaAs [eq. 11]
    EΓ0 = 1.5192  # [eV]
    EDeb = 15.9 * 1e-3  # [eV]
    ETO = 33.6 * 1e-3  # [eV]
    S = 1.8
    STO = 1.1
    EΓ_GaAs = EΓ0 + S * EDeb * (1 - coth(EDeb / V_T)) + STO * ETO * (1 - coth(ETO / V_T))  # [eV]

    # Wavelength-independent contribution to Sellmeier equation [Tab. II, GaAs FIT 2]
    A0 = 5.9613 + 7.7178e-4 * T - 0.953e-6 * T**2
    A = A0 - 16.159 * x + 43.511 * x**2 - 71.317 * x**3 + 57.535 * x**4 - 17.451 * x**5
    E1_2 = 4.7171 - 3.237e-4 * T - 1.358e-6 * T**2  # [µm⁻²]
    E2 = 0.724e-3  # [µm⁻²], GaAs E₂²
    C2 = 1.55e-3  # [µm⁻²], GaAs C₂
    E3 = 1.331e-3  # [µm⁻²], AlAs E₂²
    C3 = 2.61e-3  # [µm⁻²], AlAs C₂

    # Wavelength-dependent contributions to Sellmeier equation [Tab. IV]
    E0 = (EΓ_GaAs / µm + 1.1308 * x + 0.1436 * x**2)**2  # [µm⁻²]
    C0 = 1 / (50.535 - 150.7 * x - 62.209 * x**2 + 797.16 * x**3 - 1125 * x**4 + 503.79 * x**5)  # [µm⁻²]
    E1 = E1_2 + 11.006 * x - 3.08 * x**2  # [µm⁻²]
    C1 = 21.5647 + 113.74 * x - 122.5 * x**2 + 108.401 * x**3 - 47.318 * x**4  # [µm⁻²]

    # Sellmeier equation [eq. 12, 13]
    R = (1 - x) * C2 / (E2 - E**2) + x * C3 / (E3 - E**2)
    n2 = A + C0 / (E0 - E**2) + C1 / (E1 - E**2) + R
    return np.sqrt(n2, dtype=complex)


def nAlGaAs_Afromowitz(x, λ):
    """Al(x)Ga(1-x)As refractive index model according to [1].

    Parameters
    ----------
    x : float
        Al molar fraction (0 <= x <= 1)
    λ : float
        Wavelength in [m] (λ > 0)

    Returns
    -------
    n + ik : complex
        Complex refractive index

    References
    ----------
    .. [1] M. A. Afromowitz, "Refractive index of Ga$_{1-x}$Al$_{x}$As,"
       Solid State Communications 15, 59–63 (1974).
    """
    E = h * c / q / λ  # [eV]
    τ = 2 * np.pi

    E0 = 3.65 + 0.871 * x + 0.179 * x**2
    Ed = 36.1 - 2.45 * x
    EΓ = 1.424 + 1.266 * x + 0.26 * x**2

    E2 = E**2
    Ef2 = 2 * E0**2 - EΓ**2
    η = np.pi * Ed / (2 * E0**3 * (E0**2 - EΓ**2))

    M1 = η / τ * (Ef2**2 - EΓ**4)
    M3 = 2 * η / τ * (Ef2 - EΓ**2)
    Y = (Ef2 - E2) / (EΓ**2 - E2)
    Χ = M1 + M3 * E2 + 2 * η / τ * E2**2 * np.sign(Y) * np.log(np.abs(Y))
    n2 = 1 + Χ
    return np.sqrt(n2, dtype=complex)


def nAlGaAs_Adachi(x, λ):
    A = 6.3 + 19.0 * x
    B = 9.4 - 10.2 * x
    E0 = 1.425 + 1.155 * x + 0.37 * x**2
    Δ0 = 1.765 - 1.425 + 1.115 * x - 1.155 * x
    Χ = h * c / λ / E0 / q
    Χs = h * c / λ / (E0 + Δ0) / q

    _1st = (2 - np.sqrt(1 + Χ, dtype=complex) - np.sqrt(1 - Χ, dtype=complex)) / Χ**2
    _2nd = (2 - np.sqrt(1 + Χs, dtype=complex) - np.sqrt(1 - Χs, dtype=complex)) / Χs**2

    n = np.sqrt(A * (_1st + 0.5 * _2nd * (E0 / (E0 + Δ0))**1.5) + B, dtype=complex)
    return n


def nAlGaAs_Kim(x, λ_, Temperature=26):
    λ = λ_ * 1e6
    C = (0.52886 - 0.735 * x)**2 * (x <= 0.36) + (0.30386 - 0.105 * x)**2 * (x >= 0.36)
    A = 10.906 - 2.92 * x + 0.97501 / (λ**2 + C) + 0.002467 * (1.41 * x + 1) * λ**2
    B = (Temperature - 26) * (2.04 - 0.3 * x) * 1e-4
    n = np.sqrt(A, dtype=complex) + B
    return n


def nSiO2_Malitson(λ_):
    # I. H. Malitson, J. Opt. Soc. Am. 55, 1205-1208 (1965)
    # C. Z. Tan, J. Non-Cryst. Solids 223, 158-163 (1998)
    λ = λ_ * 1e6
    A0 = 1
    B1 = 0.6961663 / (1 - (0.0684043 / λ)**2)
    B2 = 0.4079426 / (1 - (0.1162414 / λ)**2)
    B3 = 0.8974794 / (1 - (9.896161 / λ)**2)
    n = np.sqrt(A0 + B1 + B2 + B3, dtype=complex)
    return n


def nSiO2_Arosa(λ_):
    # Y. Arosa and R. de la Fuente, Opt. Lett. 45, 4268-4271 (2020)
    λ = λ_ * 1e6
    A0 = 1
    B1 = 0.9310 / (1 - (0.079 / λ)**2)
    B2 = 0.1735 / (1 - (0.130 / λ)**2)
    B3 = 2.1121 / (1 - (14.918 / λ)**2)
    n = np.sqrt(A0 + B1 + B2 + B3, dtype=complex)
    return n


def nSi3N4_Luke(λ_):
    # K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson, Opt. Lett. 40, 4823-4826 (2015)
    λ = λ_ * 1e6
    A0 = 1
    B1 = 3.0249 / (1 - (0.1353406 / λ)**2)
    B2 = 40314 / (1 - (1239.842 / λ)**2)
    n = np.sqrt(A0 + B1 + B2, dtype=complex)
    return n


def n_ordinary_LiNbO3_Zelmon(λ_):
    λ = λ_ * 1e6
    A0 = 1
    B1 = 2.6734 * λ**2 / (λ**2 - 0.01764)
    B2 = 1.2290 * λ**2 / (λ**2 - 0.05914)
    B3 = 12.614 * λ**2 / (λ**2 - 474.60)
    n_o = np.sqrt(A0 + B1 + B2 + B3, dtype=complex)
    return n_o


def n_extraordinary_LiNbO3_Zelmon(λ_):
    λ = λ_ * 1e6
    A0 = 1
    B1 = 2.9804 * λ**2 / (λ**2 - 0.02047)
    B2 = 0.5981 * λ**2 / (λ**2 - 0.06660)
    B3 = 8.9543 * λ**2 / (λ**2 - 416.08)
    n_e = np.sqrt(A0 + B1 + B2 + B3, dtype=complex)
    return n_e
