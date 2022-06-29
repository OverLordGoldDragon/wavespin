from .scattering1d.frontend.torch_frontend import (
    ScatteringTorch1D as Scattering1D,
    TimeFrequencyScatteringTorch1D as TimeFrequencyScattering1D)

Scattering1D.__module__ = 'wavespin.torch'
Scattering1D.__name__ = 'Scattering1D'

TimeFrequencyScattering1D.__module__ = 'wavespin.torch'
TimeFrequencyScattering1D.__name__ = 'TimeFrequencyScattering1D'

__all__ = ['Scattering1D', 'TimeFrequencyScattering1D']
