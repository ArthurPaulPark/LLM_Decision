# torchvision Mocking (transforms, functional, io) to bypass 구버전 torchvision 0.2.0 import bug
from types import ModuleType

# 1. torchvision.transforms.functional.pil_to_tensor Mocking
try:
    import torchvision.transforms.functional as tv_func
    if not hasattr(tv_func, "pil_to_tensor"):
        tv_func.pil_to_tensor = lambda *args, **kwargs: None
except ImportError:
    func_mock = ModuleType("torchvision.transforms.functional")
    func_mock.pil_to_tensor = lambda *args, **kwargs: None
    sys.modules["torchvision.transforms.functional"] = func_mock

# 2. torchvision.transforms.InterpolationMode Mocking
try:
    import torchvision.transforms as tv_trans
    if not hasattr(tv_trans, "InterpolationMode"):
        class MetaInterpolationMode(type):
            def __getattr__(cls, name):
                return name.lower()
        class DummyInterpolationMode(metaclass=MetaInterpolationMode):
            NEAREST = "nearest"
            BILINEAR = "bilinear"
            BICUBIC = "bicubic"
            NEAREST_EXACT = "nearest_exact"
        tv_trans.InterpolationMode = DummyInterpolationMode
except ImportError:
    trans_mock = ModuleType("torchvision.transforms")
    class MetaInterpolationMode(type):
        def __getattr__(cls, name):
            return name.lower()
    class DummyInterpolationMode(metaclass=MetaInterpolationMode):
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        NEAREST_EXACT = "nearest_exact"
    trans_mock.InterpolationMode = DummyInterpolationMode
    sys.modules["torchvision.transforms"] = trans_mock

# 3. torchvision.io (ImageReadMode, decode_image) Mocking
try:
    from torchvision.io import ImageReadMode, decode_image
except ImportError:
    import torchvision
    io_mock = ModuleType("torchvision.io")
    io_mock.ImageReadMode = object
    io_mock.decode_image = lambda *args, **kwargs: None
    sys.modules["torchvision.io"] = io_mock
    if not hasattr(torchvision, "io"):
        torchvision.io = io_mock

# 4. torchvision.transforms.v2 & v2.functional Mocking
try:
    import torchvision.transforms.v2
except ImportError:
    v2_mock = ModuleType("torchvision.transforms.v2")
    sys.modules["torchvision.transforms.v2"] = v2_mock
    
    v2_func_mock = ModuleType("torchvision.transforms.v2.functional")
    sys.modules["torchvision.transforms.v2.functional"] = v2_func_mock
    
    v2_mock.functional = v2_func_mock
