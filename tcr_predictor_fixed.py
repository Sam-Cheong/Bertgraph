import os
import sys
import importlib.util
from types import ModuleType

# 尝试从无空格目录导入（若你把 "new train" 重命名为 "new_train"）
try:
    from new_train.train01 import TCRPredictor as _TCRPredictor  # type: ignore
except Exception:
    _TCRPredictor = None

def _load_from_path(py_path: str, mod_name: str) -> ModuleType | None:
    if not os.path.exists(py_path):
        return None
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
        return module
    except Exception:
        return None

if _TCRPredictor is None:
    # 动态尝试从 "new train/train01.py" 或 "new train/train02.py" 加载
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand_paths = [
        os.path.join(repo_root, "new_train", "train01.py"),
        os.path.join(repo_root, "new_train", "train02.py"),
    ]
    _TCRPredictor = None
    for idx, p in enumerate(cand_paths):
        m = _load_from_path(p, f"_dynamic_train{idx+1}")
        if m and hasattr(m, "TCRPredictor"):
            _TCRPredictor = getattr(m, "TCRPredictor")
            break

if _TCRPredictor is None:
    raise ImportError(
        "Cannot locate TCRPredictor. Consider renaming 'new train' to 'new_train', "
        "or ensure train01.py/train02.py defines class TCRPredictor."
    )

# 对外只暴露同名类
TCRPredictor = _TCRPredictor
__all__ = ["TCRPredictor"]