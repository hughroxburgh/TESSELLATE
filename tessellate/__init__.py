import os as _os

# Forked worker pools (asteroid prediction, per-cut asteroid writes): conda numpy's Intel OpenMP
# runtime pins every forked process to the same core -- measured on ozstar, 8 workers ran at 1.0x,
# all on one core; with KMP_AFFINITY=none they ran at 7.6x. 'none' only stops the core binding, so
# MKL's own threading is unchanged. ('disabled' makes the runtime assert and hang the workers.)
# Must be set before numpy loads, hence here.
_os.environ.setdefault("KMP_AFFINITY", "none")

# Shared data: everything a tessellate process writes (and the SLURM jobs it submits, which inherit this) is
# group read/write -- files 664, folders 775 with the usual umask 022 -- so collaborators in the project group can
# redo each other's cuts. The group itself comes from the data folders (tools._Share_folder: setgid).
_os.umask(_os.umask(0) & ~0o060)

def __getattr__(name):
    if name == "Detector":
        from .detector import Detector
        return Detector
    elif name == "Tessellate":
        from .tessellate import Tessellate
        return Tessellate
    elif name == "DataProcessor":
        from .dataprocessor import DataProcessor
        return DataProcessor
    elif name == "TessTransient":
        from .tesstransient import TessTransient
        return TessTransient
    elif name == "Navigator":
        from .navigator import Navigator
        return Navigator
    elif name == "SourceInjector":
        from .source_inject import SourceInjector
        return SourceInjector
    raise AttributeError(f"module 'tessellate' has no attribute '{name}'")