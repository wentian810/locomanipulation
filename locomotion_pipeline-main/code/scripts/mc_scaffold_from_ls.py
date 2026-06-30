#!/usr/bin/env python3
"""基于 `mc ls` 结果创建本地目录骨架。

本脚本不会复制文件内容，只做三件事：
1. 使用 `mc ls --recursive --json` 扫描远程路径。
2. 在本地创建对应目录结构。
3. 在每个本地目录写入文件列表和文件数量统计。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Optional


@dataclass
class ScanResult:
    files_by_dir: Dict[str, List[str]]
    all_dirs: Set[str]
    total_files: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List all entries under a remote mc path, create local folder structure, "
            "and save per-folder file stats. "
            "Optionally download files matching a regex pattern."
        )
    )
    parser.add_argument("remote_path", help="Remote path, e.g. syna/bucket/prefix")
    parser.add_argument("local_root", help="Local root path to create structure under")
    parser.add_argument(
        "--mc-bin",
        default="mc",
        help="mc executable to use (default: mc)",
    )
    parser.add_argument(
        "--file-list-name",
        default="_mc_files.txt",
        help="Per-folder file list filename (default: _mc_files.txt)",
    )
    parser.add_argument(
        "--file-count-name",
        default="_mc_file_count.txt",
        help="Per-folder file count filename (default: _mc_file_count.txt)",
    )
    parser.add_argument(
        "--summary-name",
        default="_mc_summary.json",
        help="Summary filename under local root (default: _mc_summary.json)",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="仅扫描目录结构，不进行下载。Only scan, skip download.",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="跳过扫描，直接从本地现有清单进行下载。Skip remote scan, download from existing local lists.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="启用文件下载模式（需配合 --regex 使用）。Enable file download mode (requires --regex).",
    )
    parser.add_argument(
        "--regex",
        type=str,
        default=None,
        help="正则表达式，匹配需要下载的文件名。Regex pattern to match files for download.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="并发下载的最大线程数（默认：4）。Max concurrent download threads (default: 4).",
    )
    return parser.parse_args()


def run_mc_ls_json(mc_bin: str, remote_path: str) -> ScanResult:
    # 通过 mc 的 JSON 行输出做结构化解析，便于后续稳定处理。
    cmd = [mc_bin, "ls", "--recursive", "--json", remote_path]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cannot find mc binary: {mc_bin}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "mc ls failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {exc.stderr.strip()}"
        ) from exc

    files_by_dir: Dict[str, List[str]] = defaultdict(list)
    all_dirs: Set[str] = {"."}
    total_files = 0

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            # 忽略非 JSON 行，避免偶发输出导致脚本中断。
            continue

        key = str(item.get("key", "")).strip()
        if not key:
            continue

        item_type = item.get("type")

        # 统一路径分隔符并去掉结尾斜杠，保证路径处理一致。
        key = key.replace("\\", "/").rstrip("/")
        if not key:
            continue

        if item_type == "folder":
            all_dirs.add(key)
            add_parent_dirs(all_dirs, key)
            continue

        # 除 folder 外，其余类型都按文件处理。
        parent, filename = split_parent_and_name(key)
        if not filename:
            continue

        files_by_dir[parent].append(filename)
        all_dirs.add(parent)
        add_parent_dirs(all_dirs, parent)
        total_files += 1

    # 排序确保输出稳定，便于比对与复现。
    for folder in files_by_dir:
        files_by_dir[folder].sort()

    return ScanResult(files_by_dir=files_by_dir, all_dirs=all_dirs, total_files=total_files)


def split_parent_and_name(path_key: str) -> tuple[str, str]:
    if "/" not in path_key:
        return ".", path_key
    parent, name = path_key.rsplit("/", 1)
    return (parent or ".", name)


def add_parent_dirs(all_dirs: Set[str], directory: str) -> None:
    if directory in ("", "."):
        all_dirs.add(".")
        return

    parts = directory.split("/")
    current = []
    for part in parts:
        if not part:
            continue
        current.append(part)
        all_dirs.add("/".join(current))


def load_scan_result_from_local(
    local_root: Path,
    file_list_name: str,
) -> ScanResult:
    """从本地现有的文件清单文件重建 ScanResult。
    
    如果本地目录已经有 _mc_files.txt，可以直接读取而无需远程扫描。
    Load ScanResult from existing local file lists without remote scan.
    """
    files_by_dir: Dict[str, List[str]] = defaultdict(list)
    all_dirs: Set[str] = set()
    total_files = 0

    if not local_root.exists():
        raise RuntimeError(f"Local root does not exist: {local_root}")

    # 遍历本地所有目录
    for dir_path in sorted(local_root.rglob(".")):
        if not dir_path.is_dir():
            continue

        # 计算相对于 local_root 的路径
        try:
            rel_dir = dir_path.relative_to(local_root)
        except ValueError:
            continue

        rel_dir_str = str(rel_dir).replace("\\", "/")
        if rel_dir_str == ".":
            rel_dir_str = "."

        # 读取该目录下的 _mc_files.txt
        file_list_path = dir_path / file_list_name
        if file_list_path.exists():
            all_dirs.add(rel_dir_str)
            content = file_list_path.read_text(encoding="utf-8").strip()
            if content:
                files = content.split("\n")
                files = [f.strip() for f in files if f.strip()]
                files_by_dir[rel_dir_str] = files
                total_files += len(files)

    # 确保至少包含根目录
    if "." not in all_dirs:
        all_dirs.add(".")

    return ScanResult(files_by_dir=files_by_dir, all_dirs=all_dirs, total_files=total_files)


def download_matching_files(
    mc_bin: str,
    remote_path: str,
    local_root: Path,
    scan: ScanResult,
    regex_pattern: str,
    max_workers: int = 4,
) -> tuple[int, int]:
    """根据正则表达式下载匹配的文件。
    
    返回 (成功下载数, 失败数)
    Compile regex pattern and download matching files from remote to local.
    Returns (downloaded_count, failed_count)
    """
    try:
        pattern = re.compile(regex_pattern)
    except re.error as exc:
        raise RuntimeError(f"Invalid regex pattern: {regex_pattern}\n{exc}") from exc

    print(f'download from remote path: {remote_path}')

    # 收集所有匹配的文件任务列表
    download_tasks: List[tuple[str, str, str]] = []
    
    for rel_dir, files in scan.files_by_dir.items():
        target_local_dir = local_root if rel_dir == "." else local_root / rel_dir
        
        for filename in files:
            # 构造完整的相对路径（相对于 remote_path）来进行正则匹配
            # 支持正则表达式包含路径部分，例如：batch1/jiashen/.*\.zip$
            if rel_dir == ".":
                rel_path = filename
            else:
                rel_path = f"{rel_dir}/{filename}"

            # 应该检查已经下载到本地的文件，如果已经存在就跳过下载，避免重复下载。
            local_file_path = target_local_dir / filename
            if local_file_path.exists():
                print(f"文件已存在，跳过下载: {local_file_path}")
                continue
            
            # 对完整的相对路径进行正则表达式匹配
            if pattern.search(rel_path):
                # 构造远程路径用于下载
                remote_file = f"{remote_path.rstrip('/')}/{rel_path}"
                # 本地目标文件的完整路径
                target_local_path = target_local_dir / filename
                # 任务中存储：远程路径、本地路径、显示用的相对路径
                download_tasks.append((remote_file, str(target_local_path), rel_path))
    
    if not download_tasks:
        print(f"未找到与正则表达式 '{regex_pattern}' 匹配的文件。")
        print(f"No files matched regex pattern '{regex_pattern}'.")
        return 0, 0
    
    print(f"\n将下载 {len(download_tasks)} 个匹配的文件...")
    print(f"Will download {len(download_tasks)} matching files...")
    
    # 并发下载
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    downloaded = 0
    failed = 0
    failed_files = []
    
    def download_file(remote_src: str, local_target: str, display_name: str) -> bool:
        """下载单个文件。Returns True if successful.
        
        Args:
            remote_src: 远程文件完整路径
            local_target: 本地目标文件的完整路径
            display_name: 用于显示的相对路径
        """
        # 确保本地目录存在
        Path(local_target).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [mc_bin, "cp", remote_src, local_target]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            print(f"Failed to download {display_name}: {exc.stderr.strip()}", file=sys.stderr)
            return False
        except Exception as exc:
            print(f"Error downloading {display_name}: {exc}", file=sys.stderr)
            return False
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, remote_src, local_target, display_name): (remote_src, display_name)
            for remote_src, local_target, display_name in download_tasks
        }
        
        for future in as_completed(futures):
            remote_src, display_name = futures[future]
            try:
                if future.result():
                    downloaded += 1
                    print(f"✓ {display_name}")
                else:
                    failed += 1
                    failed_files.append(remote_src)
            except Exception as exc:
                failed += 1
                failed_files.append(remote_src)
                print(f"Error: {exc}", file=sys.stderr)
    
    print(f"\n下载完成: {downloaded} 成功, {failed} 失败")
    print(f"Download complete: {downloaded} succeeded, {failed} failed")
    
    if failed_files:
        print("\n失败的文件列表 (Failed files):")
        for f in failed_files:
            print(f"  - {f}")
    
    return downloaded, failed


def write_outputs(
    local_root: Path,
    scan: ScanResult,
    file_list_name: str,
    file_count_name: str,
    summary_name: str,
    remote_path: str,
) -> None:
    local_root.mkdir(parents=True, exist_ok=True)

    # 先把扫描到的目录都创建出来。
    for rel_dir in sorted(scan.all_dirs):
        target_dir = local_root if rel_dir == "." else local_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

    # 在每个目录下写入文件清单和文件数量。
    for rel_dir in sorted(scan.all_dirs):
        target_dir = local_root if rel_dir == "." else local_root / rel_dir
        names = scan.files_by_dir.get(rel_dir, [])

        list_path = target_dir / file_list_name
        count_path = target_dir / file_count_name

        list_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        count_path.write_text(f"{len(names)}\n", encoding="utf-8")

    summary = {
        "remote_path": remote_path,
        "local_root": str(local_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_directories": len(scan.all_dirs),
        "total_files": scan.total_files,
        "files_listed_command": "mc ls --recursive --json",
    }
    (local_root / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    local_root = Path(args.local_root).expanduser().resolve()
    
    # 参数校验
    if args.scan_only and args.skip_scan:
        print("ERROR: --scan-only 和 --skip-scan 不能同时使用", file=sys.stderr)
        print("ERROR: --scan-only and --skip-scan are mutually exclusive", file=sys.stderr)
        return 1
    
    if args.download and not args.regex:
        print("ERROR: --download 需要搭配 --regex 使用", file=sys.stderr)
        print("ERROR: --download requires --regex", file=sys.stderr)
        return 1
    
    if args.skip_scan and args.download and not local_root.exists():
        print("ERROR: --skip-scan 模式下本地目录必须已存在", file=sys.stderr)
        print("ERROR: --skip-scan requires local directory to already exist", file=sys.stderr)
        return 1

    try:
        scan = None
        
        # 模式1：跳过扫描，直接从本地读取清单（用于下载）
        if args.skip_scan:
            print("=" * 50)
            print("从本地现有清单读取文件信息")
            print("Loading file info from local lists")
            print("=" * 50)
            try:
                scan = load_scan_result_from_local(local_root, args.file_list_name)
                print(f"\n✓ 读取完成 (Load complete)")
                print(f"  本地目录 (Local):  {local_root}")
                print(f"  目录数 (Dirs):     {len(scan.all_dirs)}")
                print(f"  文件数 (Files):    {scan.total_files}")
            except RuntimeError as err:
                print(f"ERROR: {err}", file=sys.stderr)
                return 1
        
        # 模式2和3：扫描远程路径
        else:
            print("=" * 50)
            print("第一步：扫描远程路径并创建本地目录骨架")
            print("Step 1: Scan remote path and create local structure")
            print("=" * 50)
            
            scan = run_mc_ls_json(args.mc_bin, args.remote_path)
            write_outputs(
                local_root=local_root,
                scan=scan,
                file_list_name=args.file_list_name,
                file_count_name=args.file_count_name,
                summary_name=args.summary_name,
                remote_path=args.remote_path,
            )
            
            print(f"\n✓ 扫描完成 (Scan complete)")
            print(f"  远程路径 (Remote): {args.remote_path}")
            print(f"  本地目录 (Local):  {local_root}")
            print(f"  目录数 (Dirs):     {len(scan.all_dirs)}")
            print(f"  文件数 (Files):    {scan.total_files}")
        
        # 如果是仅扫描模式，到此为止
        if args.scan_only:
            print("\n" + "=" * 50)
            print("✓ 扫描完成 (Scan only complete)")
            print("=" * 50)
            return 0
        
        # 模式1或3：如果启用下载，按正则表达式下载匹配的文件
        if args.download and args.regex:
            print("\n" + "=" * 50)
            print("第二步：按正则表达式下载匹配的文件")
            print("Step 2: Download files matching regex pattern")
            print("=" * 50)
            print(f"正则表达式 (Regex): {args.regex}")
            
            downloaded, failed = download_matching_files(
                mc_bin=args.mc_bin,
                remote_path=args.remote_path,
                local_root=local_root,
                scan=scan,
                regex_pattern=args.regex,
                max_workers=args.max_workers,
            )
            
            if failed > 0:
                return 1
        
    except RuntimeError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("\n" + "=" * 50)
    print("✓ 所有任务完成 (All tasks completed)")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
