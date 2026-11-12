"""Import every required package, print versions, and run ``pip check``.

python scripts/check_dependencies.py
"""

from __future__ import annotations

import sys

from roadmarking.healthcheck import PACKAGES, _run, check_packages


def main() -> int:
    r = _run("packages", check_packages)
    essential = {dist for _, dist, ess in PACKAGES if ess}
    for dist, ver in sorted(r.details.get("versions", {}).items(), key=lambda kv: kv[0].lower()):
        tag = "essential" if dist in essential else "optional "
        print(f"  {tag}  {dist:<26} {ver}")
    for key in ("failed_essential", "failed_optional"):
        for item in r.details.get(key, []):
            print(f"  {key}: {item}")
    print(f"pip check: {r.details.get('pip_check', '')}")
    print(f"[{r.status.upper()}] packages{' - ' + r.message if r.message else ''}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
