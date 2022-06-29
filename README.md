# WaveSpin-DAFx2022

This branch contains code used to reproduce the [DAFx2022 JTFS paper](https://arxiv.org/abs/2204.08269).

```
pip install git+https://github.com/OverLordGoldDragon/wavespin.git@dafx2022-jtfs
```

## References

WaveSpin originated as a fork of [Kymatio](https://github.com/kymatio/kymatio/) [1]. The library is showcased in [2] for audio classification and synthesis. JTFS was introduced in [3], and Wavelet Scattering in [4].

 1. M. Andreux, T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. Andén, E. Belilovsky, J. Bruna, V. Lostanlen, M. J. Hirn, E. Oyallon, S. Zhang, C. Cella, M. Eickenberg (2019). [Kymatio: Scattering Transforms in Python](https://arxiv.org/abs/1812.11214).
 2. J. Muradeli, C. Vahidi, C. Wang, H. Han, V. Lostanlen, M. Lagrange, G. Fazekas (2022). [Differentiable Time-Frequency Scattering on GPU](https://arxiv.org/abs/2204.08269).
 3. J. Anden, V. Lostanlen, S. Mallat (2015). [Joint time-frequency scattering for audio classification](https://ieeexplore.ieee.org/abstract/document/7324385).
 4. S. Mallat (2012). [Group Invariant Scattering](https://arxiv.org/abs/1101.2286).

## License

WaveSpin is MIT licensed, as found in the LICENSE file. Some source functions may be under other authorship/licenses; see NOTICE.txt.
