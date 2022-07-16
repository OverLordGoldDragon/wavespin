# WaveSpin | Differentiable Time-Frequency Scattering on GPU

This branch contains code used to reproduce the paper [Differentiable Time-Frequency Scattering on GPU](https://arxiv.org/abs/2204.08269).

The `kymatio/` code in the [experiments repository](https://github.com/cyrusvahidi/jtfs-gpu) is an older version that does not reproduce the paper, with significant differences for classification. It is also being hosted against my consent. Refer to below instructions.

## Installation

```
pip install git+https://github.com/OverLordGoldDragon/wavespin.git@dafx2022-jtfs
```

installs the algorithm. To run the experiments, clone the [experiments repository](https://github.com/cyrusvahidi/jtfs-gpu), delete `jtfs-gpu/kymatio/`, and replace `kymatio` imports with `wavespin` in `.py` files. Below is a step-by-step, with a script to automate the replacing.

### Installing experiments

 1. Install [git-bash](https://git-scm.com/downloads)
 2. Open `git-bash`. In Windows, search git-bash.
 3. `cd folder_path`, where `folder_path` is path to the folder you want to clone into. Press `Shift + Insert` in `git-bash` on Windows to paste the path. For example, `cd C:\Desktop\my_clone_dir`.
 
<img src="https://user-images.githubusercontent.com/16495490/179361976-c27a2baf-cd3e-4753-8018-9e8f711dae42.png" height="80">

 4. `git clone https://github.com/cyrusvahidi/jtfs-gpu`
 5. `cd jtfs-gpu`
 6. `git checkout 1dd4b39`. This gets the last version known to work, in case future changes break these steps. Ignore "detached HEAD".
 7. In Python, run
 
 ```python
jtfs_gpu_path = r"C:\Desktop\jtfs-gpu"  # replace with yours (step 3 + `jtfs-gpu`)
import wavespin
wavespin.fix_experiments(jtfs_gpu_path)
```

### Installing dependencies

 1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
 2. Open Anaconda Powershell Prompt (search in Windows), run below

```
conda install mamba  # much faster downloads
conda create -n jgpu-env  # virtual env, recommended
conda activate jgpu-env

cd "C:\Desktop\jtfs-gpu"  # replace with yours
mamba env update --file conda-env.yml
pip install -r pip-env.txt

pip install -e git+https://github.com/mathieulagrange/doce.git@3ad246067c6a8ac829899e7e888f4debbad80629#egg=doce
pip install git+https://github.com/PyTorchLightning/metrics.git@3af729508289d3babf0e166d9e8405cb2b0758a2
```

## Running experiments

After completing above steps, run `.py` files in `jtfs-gpu/scripts`. I highly recommend using an IDE, like [Spyder](https://github.com/spyder-ide/spyder), which is included in above installation - simply enter `spyder` in Anaconda Powershell Prompt.

Install the [dataset](https://zenodo.org/record/3464194) into `scripts` as `scripts/import/c4dm-datasets/...`. The rest of instructions can be followed reliably from the [experiments repository](https://github.com/cyrusvahidi/jtfs-gpu).

## References

WaveSpin originated as a fork of [Kymatio](https://github.com/kymatio/kymatio/) [1]. The library is showcased in [2] for audio classification and synthesis. JTFS was introduced in [3], and Wavelet Scattering in [4].

 1. M. Andreux, T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. Andén, E. Belilovsky, J. Bruna, V. Lostanlen, M. J. Hirn, E. Oyallon, S. Zhang, C. Cella, M. Eickenberg (2019). [Kymatio: Scattering Transforms in Python](https://arxiv.org/abs/1812.11214).
 2. J. Muradeli, C. Vahidi, C. Wang, H. Han, V. Lostanlen, M. Lagrange, G. Fazekas (2022). [Differentiable Time-Frequency Scattering on GPU](https://arxiv.org/abs/2204.08269).
 3. J. Anden, V. Lostanlen, S. Mallat (2015). [Joint time-frequency scattering for audio classification](https://ieeexplore.ieee.org/abstract/document/7324385).
 4. S. Mallat (2012). [Group Invariant Scattering](https://arxiv.org/abs/1101.2286).

## License

WaveSpin is MIT licensed, as found in the LICENSE file. Some source functions may be under other authorship/licenses; see NOTICE.txt.
