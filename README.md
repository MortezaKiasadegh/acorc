# **A**erosol **C**lassifier **O**perational **R**ange **C**alculator

**Available languages:** *Python*, *Javscript*

This Python package provides functions to calculate and visualize the operational ranges of various aerosol classifiers, including:

- Differential Mobility Analyzer (DMA)
- Aerosol Centrifuge Classifier (AAC)
- Centrifugal Particle Mass Analyzer (CPMA)
- Tandem configurations (CPMA-DMA, CPMA-AAC, AAC-DMA)

## Installation

The package requires the following Python dependencies:
- numpy
- scipy
- matplotlib

You can install these dependencies using pip:

```bash
pip install numpy scipy matplotlib
```

## Usage

The package provides several functions to calculate operational ranges for different classifiers:

### Single Classifiers

1. DMA (Differential Mobility Analyzer):
```python
from aerosol_classifiers import DMA_Trapezoidal

# Calculate DMA operational range
d_i, d_o, d_min, d_max, R_B = DMA_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, T_inp=298.15, P_inp=101325)
```

2. AAC (Aerosol Centrifuge Classifier):
```python
from aerosol_classifiers import AAC_Trapezoidal

# Calculate AAC operational range
d_i, d_o, d_min, d_max, R_t = AAC_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, T_inp=298.15, P_inp=101325)
```

3. CPMA (Centrifugal Particle Mass Analyzer):
```python
from aerosol_classifiers import CPMA_Trapezoidal

# Calculate CPMA operational range
d_i, d_o, d_min, d_max, R_m = CPMA_Trapezoidal(Q_a_inp=0.3, R_m_inp=3, rho100=1000, Dm=3, T_inp=298.15, P_inp=101325)
```

### Tandem Configurations

1. CPMA-DMA:
```python
from aerosol_classifiers import CPMA_DMA_Trapezoidal

# Calculate tandem CPMA-DMA operational range
d_i, d_o = CPMA_DMA_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, R_m_inp=10/3, rho100=1000, Dm=3, T_inp=298.15, P_inp=101325)
```

2. CPMA-AAC:
```python
from aerosol_classifiers import CPMA_AAC_Trapezoidal

# Calculate tandem CPMA-AAC operational range
d_i, d_o = CPMA_AAC_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=9, R_m_inp=30/2.48, rho100=510, Dm=2.48, T_inp=298.15, P_inp=101325)
```

3. AAC-DMA:
```python
from aerosol_classifiers import AAC_DMA_Trapezoidal

# Calculate tandem AAC-DMA operational range
d_i, d_o = AAC_DMA_Trapezoidal(Q_a_inp=0.3, Q_sh_inp=3, T_inp=298.15, P_inp=101325)
```

## Parameters

### Common Parameters
- `Q_a_inp`: Aerosol flow rate in L/min (default: 0.3 L/min)
- `T_inp`: Temperature in K (default: 298.15 K)
- `P_inp`: Pressure in Pa (default: 101325 Pa)
- `plot`: Whether to plot the operational range (default: True)

### DMA Parameters
- `Q_sh_inp`: Sheath flow rate in L/min (default: 3 L/min)

### AAC Parameters
- `Q_sh_inp`: Sheath flow rate in L/min (default: 3 L/min)

### CPMA Parameters
- `R_m_inp`: Mass resolution (default: 3)
- `rho100`: Effective density of particles with a mobility diameter of 100 nm (default: 1000)
- `Dm`: Mass-mobility exponent (default: 3)

## Returns

Each function returns:
- `d_i`: Lower boundary diameter [nm]
- `d_o`: Upper boundary diameter [nm]

## License

This project is licensed under the MIT License - see the LICENSE file for details. 



