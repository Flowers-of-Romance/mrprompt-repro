import sys
print("python :", sys.executable, sys.version.split()[0])
try:
    import torch
    print("torch  :", torch.__version__, "| cuda/hip avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device :", torch.cuda.get_device_name(0))
        print("hip    :", getattr(torch.version, "hip", None))
except Exception as e:
    print("torch  : IMPORT FAILED ->", repr(e))
try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers: IMPORT FAILED ->", repr(e))
try:
    import accelerate
    print("accelerate:", accelerate.__version__)
except Exception as e:
    print("accelerate: NOT INSTALLED ->", repr(e))
