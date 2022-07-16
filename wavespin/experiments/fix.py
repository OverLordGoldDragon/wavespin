import shutil
from pathlib import Path

def remove(path):
    p = Path(path)
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)

def rename(path_src, path_dst):
    p0, p1 = Path(path_src), Path(path_dst)
    if p0.is_dir() or p0.is_file():
        remove(p1)
        p0.rename(p1)


path_code = (
r"""import sys
if not any(r"{0}"
           in p for p in sys.path):
    sys.path.insert(0, r"{0}")
""")

def fix_experiments(jtfs_gpu_path):
    jtfs_gpu_path = str(Path(jtfs_gpu_path))
    # remove kymatio/, (to-be) duplicate dependencies
    for name in ('kymatio', 'conda-env-kymj.yml', 'pip-env-kymj.txt'):
        remove(Path(jtfs_gpu_path, name))
    rename(Path(jtfs_gpu_path, 'kymjtfs'), Path(jtfs_gpu_path, 'jtfs_gpu'))

    # kymatio -> wavespin; kymjtfs -> jtfs_gpu; insert path to jtfs_gpu
    for dir_name in ('jtfs_gpu', 'scripts'):
        for p in Path(jtfs_gpu_path, dir_name).iterdir():
            if p.suffix == '.py':
                with open(p, 'r') as f:
                    txt = f.read()
                with open(p, 'w') as f:
                    _txt = txt.replace('kymatio', 'wavespin'
                                       ).replace('kymjtfs', 'jtfs_gpu')
                    import_fix = path_code.format(jtfs_gpu_path)
                    if not _txt.startswith(import_fix):
                        _txt = import_fix + '\n' + _txt
                    f.write(_txt)

    # make dependencies available from jtfs-gpu
    import wavespin
    wdir = Path(Path(wavespin.__file__).parent, 'experiments')
    for name in ('conda-env.yml', 'pip-env.txt'):
        shutil.copy(Path(wdir, name), Path(jtfs_gpu_path, name))
