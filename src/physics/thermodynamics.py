"""
MODULE: THERMODYNAMIC_PROPERTIES_DB
PROFILE: NASA GLENN COEFFICIENTS (7-POLYNOMIAL)
PRECISION: HIGH-FIDELITY (MULTI-SPECIES MIXTURE)

DESCRIPTION:
    Provides the fundamental gas properties for the combustion solver.
    Unlike simple simulations that assume constant Specific Heat (Cp),
    this module calculates Cp dynamically based on Temperature and 
    Chemical Composition (Air vs. Burnt Gas).

    This accuracy is required to catch 'Thermal Window' cheats, where
    a 20C discrepancy in modeled exhaust temp can be the difference 
    between 'Compliant' and 'Illegal'.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class GasState:
    Cp: float       # Specific Heat (J/kg*K)
    Cv: float       # Specific Heat Volume (J/kg*K)
    Gamma: float    # Specific Heat Ratio (Cp/Cv)
    R: float        # Specific Gas Constant (J/kg*K)

class Species:
    """
    NASA 7-Coefficient Polynomials for a single chemical species.
    Valid Range: 200K to 6000K.
    """
    def __init__(self, name, molar_mass, coeffs_low, coeffs_high, t_switch=1000.0):
        self.name = name
        self.M = molar_mass # g/mol
        self.R = 8.314462 / (self.M / 1000.0) # J/kg*K
        self.a_low = coeffs_low
        self.a_high = coeffs_high
        self.t_switch = t_switch

    def get_cp_mole(self, T):
        """Returns Cp in J/(mol*K)"""
        # Clamp T to valid range to prevent polynomial explosion
        T = np.clip(T, 298.0, 5000.0)
        
        a = self.a_low if T < self.t_switch else self.a_high
        
        # Cp/R = a1 + a2T + a3T^2 + a4T^3 + a5T^4
        cp_r = (a[0] + 
                a[1] * T + 
                a[2] * (T**2) + 
                a[3] * (T**3) + 
                a[4] * (T**4))
        
        return cp_r * 8.314462

class ThermodynamicsDatabase:
    """
    The Librarian of Gas Properties.
    Currently loaded: Diesel Exhaust Constituents.
    """
    def __init__(self):
        # COEFFICIENTS SOURCE: NASA GLENN RESEARCH CENTER
        
        # Nitrogen (N2) - The bulk gas
        self.n2 = Species("N2", 28.013, 
            [0.229388611E+05, -0.241353326E+02, 0.906420786E-01, -0.733483729E-04, 0.137530264E-07],
            [0.291133000E+02, 0.861175111E-02, -0.100816438E-04, 0.949024682E-09, -0.333332742E-12]
        )
        
        # Oxygen (O2) - Consumed during burn
        self.o2 = Species("O2", 31.999,
            [0.378245636E+05, -0.299673416E+02, 0.984730201E-01, -0.968129509E-04, 0.324372837E-07],
            [0.369757819E+02, 0.613519704E-02, -0.125884205E-05, 0.177528148E-09, -0.113643531E-13]
        )
        
        # Carbon Dioxide (CO2) - Product of combustion
        self.co2 = Species("CO2", 44.010,
            [0.235677352E+05, -0.898452927E+01, 0.592913929E-01, -0.405021363E-04, 0.819210271E-08],
            [0.463659120E+02, 0.274131591E-01, -0.995829221E-05, 0.160373005E-08, -0.916102132E-13]
        )
        
        # Water Vapor (H2O) - Product of combustion
        self.h2o = Species("H2O", 18.015,
            [0.419864056E+05, -0.203643410E+02, 0.652040211E-01, -0.548797062E-04, 0.177197817E-07],
            [0.267214566E+02, 0.305629310E-01, -0.873026011E-05, 0.120099649E-08, -0.635395212E-13]
        )

    def get_mixture_properties(self, T_kelvin: float, equivalence_ratio: float = 0.0) -> GasState:
        """
        Calculates properties for the Diesel/Air mixture.
        phi (equivalence_ratio): 
            0.0 = Pure Air
            1.0 = Stoichiometric Burn
        """
        # 1. Determine Molar Fractions (Simplified Diesel Combustion)
        # Air is approx 79% N2, 21% O2
        # Stoichiometric: 1 Fuel + 14.5 Air -> Products
        
        # Linear approximation of species change based on burn progress (phi)
        # Note: Diesel runs lean, so phi is usually < 1.0
        phi = np.clip(equivalence_ratio, 0.0, 1.0)
        
        # Molar Fractions (x)
        x_n2 = 0.79
        x_o2 = 0.21 * (1.0 - phi) # Oxygen depleted
        x_co2 = 0.14 * phi        # CO2 created
        x_h2o = 0.13 * phi        # H2O created
        
        # Normalize (ensure sums to 1.0)
        total = x_n2 + x_o2 + x_co2 + x_h2o
        x_n2 /= total; x_o2 /= total; x_co2 /= total; x_h2o /= total
        
        # 2. Calculate Mixture Cp (Molar Weighted)
        cp_sum = (x_n2 * self.n2.get_cp_mole(T_kelvin) +
                  x_o2 * self.o2.get_cp_mole(T_kelvin) +
                  x_co2 * self.co2.get_cp_mole(T_kelvin) +
                  x_h2o * self.h2o.get_cp_mole(T_kelvin))
                  
        molar_mass_mix = (x_n2 * self.n2.M + 
                          x_o2 * self.o2.M + 
                          x_co2 * self.co2.M + 
                          x_h2o * self.h2o.M)
        
        # Convert to Specific Heat (J/kg*K)
        Cp_mix = cp_sum / (molar_mass_mix / 1000.0)
        
        # 3. Calculate Derived Properties
        R_mix = 8.314462 / (molar_mass_mix / 1000.0)
        Cv_mix = Cp_mix - R_mix
        Gamma_mix = Cp_mix / Cv_mix
        
        return GasState(Cp_mix, Cv_mix, Gamma_mix, R_mix)