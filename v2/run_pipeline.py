#!/usr/bin/env python3
"""
PocketVibe V2 — 全流程调度脚本（并发版）
自动并行启动 01a_parallel / 01b_parallel / 01c，
三者全部完成后串行跑 01d → 01e → 02
"""
import subprocess, time, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(BASE, "scripts")
DATA = os.path.join(BASE, "data", "processed")

def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

def run_script(name, label):
    """在子进程运行脚本，实时打印输出"""
    script = os.path.join(SCRIPTS, name)
    print(f"\n{'='*60}\n>>> 启动 {label} ...\n{'='*60}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE,
        env=env,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[警告] {label} 退出码={result.returncode}，继续下一步")
    else:
        print(f"[OK] {label} 完成")
    return result.returncode

def wait_for_all_three():
    """等待 01a / 01b / 01c 全部完成（根据目标行数判断）"""
    targets = {
        "multi_impl.jsonl":  ("01a", 190),   # 50×4=200，允许少量失败
        "evol_tools.jsonl":  ("01b", 700),   # 150×5=750
        "cross_cat.jsonl":   ("01c",  80),   # 约100条
    }
    print("\n>>> 等待 01a / 01b / 01c 全部完成 ...")
    while True:
        all_done = True
        status = []
        for fname, (label, threshold) in targets.items():
            n = count_lines(os.path.join(DATA, fname))
            done = n >= threshold
            status.append(f"{label}:{n}{'✅' if done else '...'}")
            if not done:
                all_done = False
        print("  " + " | ".join(status), end="\r", flush=True)
        if all_done:
            print("\n  全部完成！")
            break
        time.sleep(15)

def main():
    os.makedirs(DATA, exist_ok=True)

    # ── 01a / 01b / 01c 三者并行后台启动 ──────────────────────
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)

    print(">>> 并行启动 01a_parallel (一指令×四风格) ...")
    p01a = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPTS, "01a_multi_implementation_parallel.py")],
        cwd=BASE, env=env,
        stdout=open(os.path.join(BASE, "logs", "01a_pipeline.log"), "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    print(">>> 并行启动 01b_parallel (Evol-Instruct) ...")
    p01b = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPTS, "01b_evol_instruct_tools_parallel.py")],
        cwd=BASE, env=env,
        stdout=open(os.path.join(BASE, "logs", "01b_pipeline.log"), "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    print(">>> 并行启动 01c (跨类组合) ...")
    p01c = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPTS, "01c_cross_category.py")],
        cwd=BASE, env=env,
        stdout=open(os.path.join(BASE, "logs", "01c_pipeline.log"), "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    print(f"  01a PID={p01a.pid}  01b PID={p01b.pid}  01c PID={p01c.pid}")
    print(f"  日志: logs/01a_pipeline.log / 01b_pipeline.log / 01c_pipeline.log")

    # ── 等待三者全部完成 ───────────────────────────────────────
    wait_for_all_three()

    # ── 串行后处理 ─────────────────────────────────────────────
    run_script("01d_merge_and_dedupe.py",  "01d 合并去重")
    run_script("01e_static_validate.py",   "01e 静态校验")
    run_script("02_category_split.py",     "02  类别切分")

    print("\n" + "="*60)
    print("✅ 全流程完成！")
    for fname in ["train.jsonl", "val.jsonl"]:
        n = count_lines(os.path.join(DATA, fname))
        print(f"   {fname}: {n} 条")
    print("="*60)
    print("下一步：上传到 HPC → sbatch slurm/train.slurm")

if __name__ == "__main__":
    main()
