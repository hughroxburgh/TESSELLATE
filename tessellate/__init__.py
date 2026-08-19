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