from ...frontend.entry import ScatteringEntry

class ScatteringEntry1D(ScatteringEntry):
    """
    Kymatio, (C) 2018-present. The Kymatio developers.
    https://github.com/kymatio/kymatio/blob/master/kymatio/scattering1d/frontend/
    entry.py
    """
    def __init__(self, *args, **kwargs):
        super().__init__(name='1D', class_name='scattering1d', *args, **kwargs)

class TimeFrequencyScatteringEntry1D(ScatteringEntry):
    def __init__(self, *args, **kwargs):
        super().__init__(name='1D', class_name='scattering1d', *args, **kwargs)

__all__ = ['ScatteringEntry1D', 'TimeFrequencyScatteringEntry1D']
