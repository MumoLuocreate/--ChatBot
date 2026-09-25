from __future__ import annotations
import argparse, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qichi.readiness import ReadinessError, check_ready, compute_build_id
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("marker");p.add_argument("lock");p.add_argument("--owner",required=True);p.add_argument("--bot",required=True);p.add_argument("--build-root",type=Path,default=_ROOT,help="source tree the READY marker must vouch for");a=p.parse_args(argv)
    try:expected_build_id=compute_build_id(a.build_root)
    except ReadinessError as e:print(f"READY: FAIL ({e})",file=sys.stderr);return 1
    try:check_ready(a.marker,a.lock,expected_owner=a.owner,expected_bot=a.bot,expected_build_id=expected_build_id)
    except ReadinessError as e:print(f"READY: FAIL ({e})",file=sys.stderr);return 1
    print("READY: OK");return 0
if __name__=="__main__":raise SystemExit(main())
