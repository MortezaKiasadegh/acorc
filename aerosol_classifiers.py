import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def Cc(d, P, T):
    """
    Calculates the Cunningham slip correction factor.

    Parameters
    ----------
    d : array
        Particle diameter in m.
    P : array
        Pressure(s) in Pascals.
    T : array
        Temperature(s) in Kelvin.

    Returns
    -------
    float or array-like
        Cunningham slip correction factor(s).
    """
    # Constants
    alpha = 1.165 * 2
    beta = 0.483 * 2
    gamma = 0.997 / 2

    la = 67.30e-9  # mfp of air at 101.325kPa and 296.15 K

    # Calculate mean free path of air at new P and T using NumPy broadcasting
    lap = la * (T / 296.15) ** 2 * (101325/P) * ((110.4 + 296.15) / (T+110.4))

    # Calculate Cunningham slip correction factor using NumPy broadcasting
    Cc = 1 + lap / d * (alpha + beta * np.exp(-gamma * d / lap))

    return Cc

def DMA_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, plot=True):
    """
    Calculates the DMA operational range using the Sheath and aerosol flow rates.

    Parameters
    ----------
    Q_a_inp : float
        Aerosol flow rate in L/min (default: 0.3 L/min).
    Q_sh_inp : float
        Sheath flow rate in L/min (default: 3 L/min).
    plot : bool
        Whether to plot the operational range (default: True).

    Returns
    -------
    d_i : float
        Lower boundary diameter for input Q_sh [nm].
    d_o : float
        Upper boundary diameter for input Q_sh [nm].
    d_min_DMA : ndarray
        Lower boundary diameters for Q_sh sweep [nm].
    d_max_DMA : ndarray
        Upper boundary diameters for Q_sh sweep [nm].
    R_B : ndarray
        Range of R_B values.
    """
    # Constants
    P = 101325  # The pressure in pa
    T = 298.15  # The temperature in K
    mu = 1.81809e-5 * (T / 293.15) ** 1.5 * (293.15 + 110.4) / (T + 110.4)  # viscosity of gas (see Rader (1990))
    Q_a = Q_a_inp / 60000  # The aerosol flow rate in m^3/s
    Q_sh = Q_sh_inp / 60000   # The sheath flow rate in m^3/s
    Q_sh_lb = 2 / 60000  # Lower limit of sheath flow in m^3/s
    Q_sh_ub = 30 / 60000  # Upper limit of sheath flow in m^3/s
    r_1 = 9.37e-3  # inner cylinder radius in m
    r_2 = 19.61e-3  # outer cylinder radius in m
    L = 0.44369  # length of clasifier in m
    e = 1.6e-19  # elementary charge in C
    V_min = 10  # minimum potential difference in v
    V_max = 10000  # maximum potential difference in v

    log_r_ratio = np.log(r_2 / r_1)
    factor1 = (2 * V_min * L * e) / (3 * mu * log_r_ratio)
    factor2 = (2 * V_max * L * e) / (3 * mu * log_r_ratio)

    Q_sh_spa = np.logspace(np.log10(Q_sh_lb), np.log10(Q_sh_ub), num=200)
    R_B = Q_sh_spa / Q_a
    R_B_lb = Q_sh_lb / Q_a
    R_B_up = Q_sh_ub / Q_a

    def f(d, Q_sh_val, factor):
        return d - (factor / Q_sh_val) * Cc(d, P, T)

    d_min_DMA = np.zeros_like(Q_sh_spa)
    d_max_DMA = np.zeros_like(Q_sh_spa)

    d_i = np.zeros_like(Q_sh_spa)
    d_o = np.zeros_like(Q_sh_spa)

    # Initial guess
    initial_guess_l = np.ones_like(Q_sh_spa) * 1e-8
    initial_guess_h = np.ones_like(Q_sh_spa) * 1e-6

    # Solve for d_min and d_max using fsolve
    d_min_DMA = fsolve(f, initial_guess_l, args=(Q_sh_spa, factor1))
    d_max_DMA = fsolve(f, initial_guess_h, args=(Q_sh_spa, factor2))

    d_i = fsolve(f, initial_guess_l[0], args=(Q_sh, factor1))[0]
    d_o = fsolve(f, initial_guess_h[0], args=(Q_sh, factor2))[0]

    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog([d_min_DMA[0] * 1e9, d_max_DMA[0] * 1e9], [R_B_lb, R_B_lb], color='red')
        plt.loglog([d_min_DMA[-1] * 1e9, d_max_DMA[-1] * 1e9], [R_B_up, R_B_up], color='red')
        plt.loglog(d_min_DMA * 1e9, R_B, color='red')
        plt.loglog(d_max_DMA * 1e9, R_B, color='red', label='DMA operational range')
        plt.loglog([d_i * 1e9, d_o * 1e9], [Q_sh / Q_a, Q_sh / Q_a], color='green', label=r'Input $Q_{sh}$ boundary')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.xlabel(r'Mobility diameter, $D_{\mathrm{m}}$ [nm]')
        plt.ylabel(r'$R_{\tau}$')
        plt.title('DMA Operational Range')
        plt.show()

    return (d_i * 1e9, d_o * 1e9,
            d_min_DMA * 1e9,
            d_max_DMA * 1e9,
            R_B)

def AAC_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, plot=True):
    """
    Calculates the AAC operational range using the Sheath and aerosol flow rates.

    Parameters
    ----------
    Q_a_inp : float
        Aerosol flow rate in L/min (default: 0.3 L/min).
    Q_sh_inp : float
        Sheath flow rate in L/min (default: 3 L/min).
    plot : bool
        Whether to plot the operational range (default: True).

    Returns
    -------
    d_i : float
        Lower boundary diameter for input Q_sh [nm].
    d_o : float
        Upper boundary diameter for input Q_sh [nm].
    d_min_AAC : ndarray
        Lower boundary diameters for Q_sh sweep [nm].
    d_max_AAC : ndarray
        Upper boundary diameters for Q_sh sweep [nm].
    R_t : ndarray
        Range of R_t values.
    """
    # Gas properties
    P = 101325  # The pressure in pa
    T = 298.15  # The temperature in K
    mu = 1.81809e-5 * (T / 293.15) ** 1.5 * (293.15 + 110.4) / (T + 110.4)  # viscosity of gas (see Rader (1990))

    # Flow rates
    Q_a = Q_a_inp / 60000  # The aerosol flow rate in m^3/s
    Q_sh = Q_sh_inp / 60000  # The sheath flow rate in m^3/s
    Q_sh_lb = 2 / 60000  # Lower limit of sheath flow in m^3/s
    Q_sh_ub = 15 / 60000  # Upper limit of sheath flow in m^3/s
    Q_sh_RB = 10 / 60000  # Rayleigh-Bénard sheath flow in m^3/s

    # Classifier properties
    r_1 = 56e-3  # inner cylinder radius in m
    r_2 = 60e-3  # outer cylinder radius in m
    L = 0.206  # length of clasifier in m

    # Rotational speed bounds
    w_lb_i = 2 * np.pi / 60 * 200
    w_ub_i = 2 * np.pi / 60 * 7000

    Q_sh_spa = np.logspace(np.log10(Q_sh_lb), np.log10(Q_sh_ub), num=200)
    R_t = Q_sh_spa / Q_a
    R_t_lb = Q_sh_lb / Q_a
    R_t_up = Q_sh_ub / Q_a

    w_low = np.ones_like(Q_sh_spa) * w_lb_i
    w_up = np.zeros_like(Q_sh_spa)
    for i, Q in enumerate(Q_sh_spa):
        if Q < Q_sh_RB:
            w_up[i] = np.min([w_ub_i, 723.7 - 9.87 * 60000 * Q])
        else:
            w_up[i] = np.min([w_ub_i, 875 - 25 * 60000 * Q])

    factor1 = (36 * mu)/ (np.pi * 1000 * (r_1 + r_2)**2 * L * (w_low**2))
    factor2 = (36 * mu)/ (np.pi * 1000 * (r_1 + r_2)**2 * L * (w_up**2))

    def f(d, Q_sh_val, factor_val):
        return d**2 * Cc(d, P, T) - (factor_val * Q_sh_val)

    d_min_AAC = np.zeros_like(Q_sh_spa)
    d_max_AAC = np.zeros_like(Q_sh_spa)

    # Initial guess
    initial_guess_l = np.ones_like(Q_sh_spa) * 1e-8
    initial_guess_h = np.ones_like(Q_sh_spa) * 1e-6

    # Solve for d_min and d_max using fsolve
    d_min_AAC = fsolve(f, initial_guess_l, args=(Q_sh_spa, factor2))
    d_max_AAC = fsolve(f, initial_guess_h, args=(Q_sh_spa, factor1))

    # Calculate d_i and d_o using Q_sh_input:
    # Find the index corresponding to Q_sh_input in Q_sh_spa
    index = np.argmin(np.abs(Q_sh_spa - Q_sh))

    # Use the index to find the corresponding values of factor1 and factor2
    factor1_input = factor1[index]
    factor2_input = factor2[index]

    # Solve for d_i and d_o using fsolve
    d_i = fsolve(f, initial_guess_l[0], args=(Q_sh, factor2_input))[0]
    d_o = fsolve(f, initial_guess_h[0], args=(Q_sh, factor1_input))[0]

    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog([d_min_AAC[0] * 1e9, d_max_AAC[0] * 1e9], [R_t_lb, R_t_lb], color='red')
        plt.loglog([d_min_AAC[-1] * 1e9, d_max_AAC[-1] * 1e9], [R_t_up, R_t_up], color='red')
        plt.loglog(d_min_AAC * 1e9, R_t, color='red')
        plt.loglog(d_max_AAC * 1e9, R_t, color='red', label='AAC operational range')
        plt.loglog([d_i * 1e9, d_o * 1e9], [Q_sh / Q_a, Q_sh / Q_a], color='green', label=r'Input $Q_{sh}$ boundary')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.xlabel(r'Aerodynamic diameter, $D_{\mathrm{a}}$ [nm]')
        plt.ylabel(r'$R_{\tau}$')
        plt.title('AAC Operational Range')
        plt.show()

    return (d_i * 1e9, d_o * 1e9,
            d_min_AAC * 1e9,
            d_max_AAC * 1e9,
            R_t)

def CPMA_Trapezoidal(Q_a_inp=0.3, R_m_inp=3, rho100=1000, Dm=1000, plot=True):
    """
    Calculates the AAC operational range using the Sheath and aerosol flow rates.

    Parameters
    ----------
    Q_a_inp : float
        Aerosol flow rate in L/min (default: 0.3 L/min).
    R_m_inp : float
        Mass resolution (default: 5).
    plot : bool
        Whether to plot the operational range (default: True).

    Returns
    -------
    d_i : float
        Lower boundary diameter for input R_m [nm].
    d_o : float
        Upper boundary diameter for input R_m [nm].
    d_min_CPMA : ndarray
        Lower boundary diameters for R_m sweep [nm].
    d_max_CPMA : ndarray
        Upper boundary diameters for R_m sweep [nm].
    R_m_spa : ndarray
        Range of R_m values.
    """
    # Gas properties
    P = 101325  # Pa
    T = 298.15  # K
    mu = 1.81809e-5 * (T / 293.15) ** 1.5 * (293.15 + 110.4) / (T + 110.4)

    # Flow rates
    Q_a = Q_a_inp / 60000  # m^3/s

    # Classifier parameters
    r_1 = 60e-3
    r_2 = 61e-3
    r_c = 60.5e-3
    L = 0.2
    e = 1.6e-19
    V_min = 0.1
    V_max = 1000
    w_lb_i = 2 * np.pi / 60 * 200
    w_ub_i = 2 * np.pi / 60 * 12000

    # Mass-mobility parameters
    rho_eff_100 = rho100
    D_m = Dm
    k = np.pi / 6 * rho_eff_100 * np.pow(100*1e-9, 3-D_m)
    R_m_spa = np.logspace(np.log10(0.001), np.log10(200), num=200)

    # Precompute factors
    log_r_ratio = np.log(r_2 / r_1)
    factor1 = e * V_min / (k * r_c**2 * log_r_ratio)
    factor2 = e * V_max / (k * r_c**2 * log_r_ratio)
    factor3 = 3 * mu * Q_a / (2 * k * r_c**2 * L)

    # Generic residual function
    def residual(d, R_m_val, factor_v, is_min, is_first):
        d = np.atleast_1d(d)[0]
        d_m_max = np.pow(((R_m_val + 1) / R_m_val), (1 / D_m)) * d
        factor = factor1 if factor_v == 'min' else factor2

        w_guess = (factor / np.pow(d, D_m))**0.5
        if is_first:
            w = np.clip(w_guess, w_lb_i, w_ub_i) if is_min else np.clip(w_guess, w_lb_i, w_ub_i)
        else:
            w = np.clip(w_guess, w_lb_i, w_ub_i)

        res = np.pow(d, D_m) - np.pow(d_m_max, D_m) + (factor3 / w**2) * (d_m_max / Cc(d_m_max, P, T))
        return res / np.abs(np.pow(d, D_m))

    # Solve d_i and d_o for the input R_m
    def optimize_diameter(R_m_val, factor_v, is_min, is_first, guess):
        return least_squares(
            residual, guess, bounds=(1e-9, 5e-6),
            args=(R_m_val, factor_v, is_min, is_first)
        ).x[0]

    # Single point (input R_m)
    d_i_1 = optimize_diameter(R_m_inp, 'min', True, True, 1.5e-8)
    d_o_1 = optimize_diameter(R_m_inp, 'min', False, True, 1.5e-6)
    d_i_2 = optimize_diameter(R_m_inp, 'max', True, False, 1.5e-8)
    d_o_2 = optimize_diameter(R_m_inp, 'max', False, False, 1.5e-6)

    d_i = max(d_i_1, d_i_2)
    d_o = min(d_o_1, d_o_2)

    # Allocate arrays for the R_m_spa sweep
    d_min_1, d_max_1 = np.zeros_like(R_m_spa), np.zeros_like(R_m_spa)
    d_min_2, d_max_2 = np.zeros_like(R_m_spa), np.zeros_like(R_m_spa)

    # Loop over R_m_spa with vectorized initial guesses (optional)
    for idx, R_m_val in enumerate(R_m_spa):
        d_min_1[idx] = optimize_diameter(R_m_val, 'min', True, True, 1.5e-8)
        d_max_1[idx] = optimize_diameter(R_m_val, 'min', False, True, 1.5e-6)
        d_min_2[idx] = optimize_diameter(R_m_val, 'max', True, False, 1.5e-8)
        d_max_2[idx] = optimize_diameter(R_m_val, 'max', False, False, 1.5e-6)

    # Combine results
    d_min_CPMA = np.maximum(d_min_1, d_min_2)
    d_max_CPMA = np.minimum(d_max_1, d_max_2)

    valid_indices = np.where(abs(d_min_CPMA - d_max_CPMA) > 1e-8)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.loglog(d_min_CPMA[valid_indices] * 1e9, R_m_spa[valid_indices], color='red')
        plt.loglog(d_max_CPMA[valid_indices] * 1e9, R_m_spa[valid_indices], color='red', label='CPMA operational range')
        plt.loglog([d_i * 1e9, d_o * 1e9], [R_m_inp, R_m_inp], color='green', label=r'Input $R_{\mathrm{m}}$ boundary')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.xlabel(r'Mobility diameter, $D_{\mathrm{m}}$ [nm]')
        plt.ylabel(r'$R_{\mathrm{m}}$')
        plt.title('CPMA Operational Range')
        plt.show()

    return (d_i * 1e9, d_o * 1e9,
            d_min_CPMA[valid_indices] * 1e9,
            d_max_CPMA[valid_indices] * 1e9,
            R_m_spa[valid_indices])

def CPMA_DMA_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, R_m_inp=10/3, rho100=1000, Dm=3):
    """
    Calculates the tandem CPMA-DMA operational range using the Sheath, aerosol flow rates, mass resolution and mass-mobility relationship.

    Parameters
    ----------
    Q_a_inp : float
        Aerosol flow rate in L/min (default: 0.3 L/min).
    Q_sh_inp : float
        Sheath flow rate in L/min (default: 3 L/min).
    R_m_inp : float
        Mass resolution (default: 10/3).
    rho100 : float
        Effective density of particles with a mobility diameter of 100 nm (default: 1000).
    Dm : float
        Mass-mobility exponent (default: 3).

    Returns
    -------
    d_i : float
        Lower boundary diameter.
    d_o : float
        Upper boundary diameter.
    """
    d_i_DMA, d_o_DMA, d_min_DMA, d_max_DMA, R_B = DMA_Trapezoidal(Q_a_inp, Q_sh_inp, False)

    d_i_CPMA, d_o_CPMA, d_min_CPMA, d_max_CPMA, R_m = CPMA_Trapezoidal(Q_a_inp, R_m_inp, rho100, Dm, False)

    d_i = max(d_i_DMA, d_i_CPMA)
    d_o = min(d_o_DMA, d_o_CPMA)

    Q_sh_lb = 2  # Lower limit of sheath flow
    Q_sh_ub = 30  # Upper limit of sheath flow
    R_B_lb = Q_sh_lb / Q_a_inp
    R_B_up = Q_sh_ub / Q_a_inp

    print(f'Maximum range: [{d_i}, {d_o}] nm')

    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(d_min_DMA, R_B, color='red')
    plt.loglog(d_max_DMA, R_B, color='red', label='DMA')
    plt.loglog([d_min_DMA[0], d_max_DMA[0]], [R_B_lb, R_B_lb], color='red')
    plt.loglog([d_min_DMA[-1], d_max_DMA[-1]], [R_B_up, R_B_up], color='red')

    plt.loglog(d_min_CPMA, Dm * R_m, color='Blue')
    plt.loglog(d_max_CPMA, Dm * R_m, color='Blue', label='CPMA')

    plt.loglog([d_i, d_o], [Q_sh_inp / Q_a_inp, Q_sh_inp / Q_a_inp], color='green', label=r'Input $Q_{sh}$ and $R_{\mathrm{m}}$ boundary')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlabel(r'Particle diameter, $d_{\mathrm{p}}$ [nm]')
    plt.ylabel(rf'$R_\mathrm{{B}}$, {Dm}$R_\mathrm{{m}}$')
    plt.title('Tandem CPMA-DMA Operational Range')
    plt.show()
    return d_i, d_o

def CPMA_AAC_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, R_m_inp=10/3, rho100=1000, Dm=3):
    """
    Calculates the tandem CPMA-AAC operational range using the Sheath, aerosol flow rates, mass resolution and mass-mobility relationship.

    Parameters
    ----------
    Q_a_inp : float
        Aerosol flow rate in L/min (default: 0.3 L/min).
    Q_sh_inp : float
        Sheath flow rate in L/min (default: 3 L/min).
    R_m_inp : float
        Mass resolution (default: 10/3).
    rho100 : float
        Effective density of particles with a mobility diameter of 100 nm (default: 1000).
    Dm : float
        Mass-mobility exponent (default: 3).

    Returns
    -------
    d_i : float
        Lower boundary diameter.
    d_o : float
        Upper boundary diameter.
    """
    d_i_AAC, d_o_AAC, d_min_AAC, d_max_AAC, R_t = AAC_Trapezoidal(Q_a_inp, Q_sh_inp, False)

    d_i_CPMA, d_o_CPMA, d_min_CPMA, d_max_CPMA, R_m = CPMA_Trapezoidal(Q_a_inp, R_m_inp, rho100, Dm, False)

    d_i = max(d_i_AAC, d_i_CPMA)
    d_o = min(d_o_AAC, d_o_CPMA)

    Q_sh_lb = 2  # Lower limit of sheath flow
    Q_sh_ub = 15  # Upper limit of sheath flow
    R_t_lb = Q_sh_lb / Q_a_inp
    R_t_up = Q_sh_ub / Q_a_inp

    print(f'Maximum range: [{d_i}, {d_o}] nm')

    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(d_min_AAC, R_t, color='red')
    plt.loglog(d_max_AAC, R_t, color='red', label='AAC')
    plt.loglog([d_min_AAC[0], d_max_AAC[0]], [R_t_lb, R_t_lb], color='red')
    plt.loglog([d_min_AAC[-1], d_max_AAC[-1]], [R_t_up, R_t_up], color='red')

    plt.loglog(d_min_CPMA, Dm * R_m, color='Blue')
    plt.loglog(d_max_CPMA, Dm * R_m, color='Blue', label='CPMA')

    plt.loglog([d_i, d_o], [Q_sh_inp / Q_a_inp, Q_sh_inp / Q_a_inp], color='green', label=r'Input $Q_{sh}$ and $R_{\mathrm{m}}$ boundary')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlabel(r'Particle diameter, $d_{\mathrm{p}}$ [nm]')
    plt.ylabel(rf'$R_\tau$, {Dm}$R_\mathrm{{m}}$')
    plt.title('Tandem CPMA-AAC Operational Range')
    plt.show()
    return d_i, d_o

def AAC_DMA_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3):
    """
    Calculates the tandem AAC-DMA operational range using the Sheath, aerosol flow rates.

    Parameters
    ----------
    Q_a_inp : float
        Aerosol flow rate in L/min (default: 0.3 L/min).
    Q_sh_inp : float
        Sheath flow rate in L/min (default: 3 L/min).

    Returns
    -------
    d_i : float
        Lower boundary diameter.
    d_o : float
        Upper boundary diameter.
    """
    d_i_AAC, d_o_AAC, d_min_AAC, d_max_AAC, R_t = AAC_Trapezoidal(Q_a_inp, Q_sh_inp, False)

    d_i_DMA, d_o_DMA, d_min_DMA, d_max_DMA, R_B = DMA_Trapezoidal(Q_a_inp, Q_sh_inp, False)

    d_i = max(d_i_DMA, d_i_AAC)
    d_o = min(d_o_DMA, d_o_AAC)

    Q_sh_lb_AAC = 2  # Lower limit of sheath flow
    Q_sh_ub_AAC = 15  # Upper limit of sheath flow
    R_t_lb = Q_sh_lb_AAC / Q_a_inp
    R_t_up = Q_sh_ub_AAC / Q_a_inp

    Q_sh_lb_DMA = 2  # Lower limit of sheath flow
    Q_sh_ub_DMA = 30  # Upper limit of sheath flow
    R_B_lb = Q_sh_lb_DMA / Q_a_inp
    R_B_up = Q_sh_ub_DMA / Q_a_inp

    print(f'Maximum range: [{d_i}, {d_o}] nm')

    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(d_min_AAC, R_t, color='red')
    plt.loglog(d_max_AAC, R_t, color='red', label='AAC')
    plt.loglog([d_min_AAC[0], d_max_AAC[0]], [R_t_lb, R_t_lb], color='red')
    plt.loglog([d_min_AAC[-1], d_max_AAC[-1]], [R_t_up, R_t_up], color='red')

    plt.loglog(d_min_DMA, R_B, color='Blue')
    plt.loglog(d_max_DMA, R_B, color='Blue', label='DMA')
    plt.loglog([d_min_DMA[0], d_max_DMA[0]], [R_B_lb, R_B_lb], color='Blue')
    plt.loglog([d_min_DMA[-1], d_max_DMA[-1]], [R_B_up, R_B_up], color='Blue')

    plt.loglog([d_i, d_o], [Q_sh_inp / Q_a_inp, Q_sh_inp / Q_a_inp], color='green', label=r'Input $Q_{sh}$ boundary')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlabel(r'Particle diameter, $d_{\mathrm{p}}$ [nm]')
    plt.ylabel(r'$R_{\tau}$, $R_{B}$')
    plt.title('Tandem AAC-DMA Operational Range')
    plt.show()
    return d_i, d_o 
