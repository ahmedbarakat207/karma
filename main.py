import os
import sys
import warnings

# Suppress framework & dependency warnings and C++ glog output
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

class SilenceStderrFD:
    """Temporarily silences C-level file descriptor 2 (stderr) to suppress Objective-C duplicate symbol warnings."""
    def __enter__(self):
        try:
            sys.stderr.flush()
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.saved_stderr_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "null_fd", None) is not None:
            try:
                sys.stderr.flush()
                os.dup2(self.saved_stderr_fd, 2)
                os.close(self.saved_stderr_fd)
                os.close(self.null_fd)
            except Exception:
                pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
