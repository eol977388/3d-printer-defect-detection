"""正式B0训练入口：验证abnormal binary clean_v3后调用原版train.py。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from verify_frozen_dataset import verify_dataset


PROJECT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT / "data" / "3d_printer_abnormal_binary_v3.yaml"
FREEZE_DIR = PROJECT / "reports" / "data_freeze" / "abnormal_binary_clean_v3"


def main() -> int:
    user_args = sys.argv[1:]
    if "--data" in user_args or any(arg.startswith("--data=") for arg in user_args):
        print("禁止覆盖--data：本入口固定使用data/3d_printer_abnormal_binary_v3.yaml", file=sys.stderr)
        return 2
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    dataset = (PROJECT / data["path"]).resolve()
    print("正式训练前正在验证abnormal binary clean_v3完整性……", flush=True)
    result = verify_dataset(dataset, FREEZE_DIR)
    if not result["通过"]:
        print("FAIL：clean_v3与冻结记录不一致，训练已阻止。", file=sys.stderr)
        return 1
    print(f"PASS：冻结验证通过，指纹={result['当前总指纹']}", flush=True)
    command = [sys.executable, str(PROJECT / "train.py"), "--data", str(DATA_YAML), *user_args]
    print("启动命令：", subprocess.list2cmdline(command), flush=True)
    return subprocess.call(command, cwd=PROJECT)


if __name__ == "__main__":
    raise SystemExit(main())
