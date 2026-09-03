import os
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SilenceStderrFD

with SilenceStderrFD():
    try:
        import cv2
        import av
        import torch
        import faster_whisper
    except Exception:
        pass
    from src.main import main

if __name__ == "__main__":
    main()
