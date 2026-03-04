import os
import sys
import runpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "datastates_train_bloom_generic_p_auto.py")

def norm_path(p: str) -> str:
    if not p:
        p = os.getcwd()
    return os.path.abspath(p)

# repo root가 sys.path에 있으면 local datastates가 site-packages를 가림
sys.path = [p for p in sys.path if norm_path(p) != REPO]

print("[DEBUG] running:", SCRIPT)
print("[DEBUG] cwd:", os.getcwd())
print("[DEBUG] sys.path[0]:", sys.path[0] if sys.path else None)

runpy.run_path(SCRIPT, run_name="__main__")
