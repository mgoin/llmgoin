#!/usr/bin/env python3
"""
Generate a report of the vLLM wheel size, including shared object breakdown by CUDA gencode
and Python code size.
"""
import argparse
import os
import re
import sys
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile


def detect_gencodes(so_path):
    """
    Detect CUDA gencode targets in the shared object using external utilities.
    Returns a list of strings like '7.5', '8.0', etc., or ['unknown'] if detection fails.
    """
    gencodes = set()
    nvdisasm = shutil.which('nvdisasm')
    cuobjdump = shutil.which('cuobjdump')
    # Try CUDA binary utilities to list embedded architectures
    tools = []
    if cuobjdump:
        tools.append((cuobjdump, ['--list-elf']))
        tools.append((cuobjdump, ['--list-ptx']))
    if nvdisasm:
        tools.append((nvdisasm, ['-list-sass']))
    for tool, args in tools:
        try:
            proc = subprocess.run([tool] + args + [so_path], capture_output=True, text=True, check=True)
            output = proc.stdout + proc.stderr
            # match sm_XX, compute_XX, including optional letter suffix (e.g. sm_90a)
            for m in re.finditer(r'(?:sm|compute)_([0-9]{2,3}[a-z]?)', output, re.IGNORECASE):
                tok = m.group(1)
                # separate letter suffix if present
                letter = tok[-1] if tok[-1].isalpha() else ''
                num = tok[:-1] if letter else tok
                if len(num) >= 2:
                    major = num[:-1]
                    minor = num[-1]
                    g = f"{int(major)}.{minor}{letter}"
                else:
                    g = f"{int(num)}{letter}"
                gencodes.add(g)
            if gencodes:
                # sort numeric parts, place unknown/others at end
                try:
                    g_sorted = sorted(gencodes, key=lambda x: float(re.match(r"(\d+(?:\.\d+)?)", x).group(1)))
                except Exception:
                    g_sorted = sorted(gencodes)
                return g_sorted
        except Exception:
            continue
    # Fallback: scan binary strings
    if shutil.which('strings'):
        try:
            output = subprocess.check_output(['strings', so_path], text=True, errors='ignore')
            for m in re.finditer(r'(?:sm|compute)_([0-9]{2,3}[a-z]?)', output, re.IGNORECASE):
                tok = m.group(1)
                letter = tok[-1] if tok[-1].isalpha() else ''
                num = tok[:-1] if letter else tok
                if len(num) >= 2:
                    major = num[:-1]
                    minor = num[-1]
                    g = f"{int(major)}.{minor}{letter}"
                else:
                    g = f"{int(num)}{letter}"
                gencodes.add(g)
            if gencodes:
                try:
                    g_sorted = sorted(gencodes, key=lambda x: float(re.match(r"(\d+(?:\.\d+)?)", x).group(1)))
                except Exception:
                    g_sorted = sorted(gencodes)
                return g_sorted
        except Exception:
            pass
    return ['unknown']

def analyze_wheel(path):
    """
    Analyze the wheel file at the given path and return size statistics.
    Returns a dict with keys:
      - python_files: list of (path, size)
      - so_files: list of (path, size, compressed_size, [gencodes])
      - gencode_summary: dict of gencode -> total size
      - python_total: total size of Python files
      - so_total: total size of .so files
      - total_size: total size of all files
    """
    stats = {
        'python_files': [],
        'so_files': [],
        'gencode_summary': {},
    }
    total_size = 0
    so_total = 0
    py_total = 0
    with zipfile.ZipFile(path, 'r') as z, tempfile.TemporaryDirectory() as tmpdir:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            size = info.file_size
            compressed_size = info.compress_size
            total_size += size
            if name.endswith('.py'):
                stats['python_files'].append((name, size))
                py_total += size
            elif name.endswith('.so'):
                # extract .so and measure per-gencode chunk sizes
                z.extract(info, tmpdir)
                so_path = os.path.join(tmpdir, name)
                # prepare per-ELF extraction directory
                so_dir = tempfile.mkdtemp(dir=tmpdir)
                # detect cuobjdump for extraction
                cuobjdump = shutil.which('cuobjdump')
                so_gencode_sizes = {}
                if cuobjdump:
                    # extract all embedded ELFs (cubin/PTX) to so_dir
                    try:
                        subprocess.run([cuobjdump, '--extract-elf', 'all', so_path],
                                       cwd=so_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        # iterate extracted files
                        for fn in os.listdir(so_dir):
                            if fn.endswith('.cubin') or fn.endswith('.ptx') or fn.endswith('.o'):
                                # parse gencode from filename e.g. '*.sm_80.cubin'
                                m = re.search(r'sm_([0-9]{2,3})([a-z]?)', fn, re.IGNORECASE)
                                if m:
                                    num, letter = m.group(1), m.group(2)
                                    major = num[:-1] if len(num) > 1 else num
                                    minor = num[-1]
                                    gcode = f"{int(major)}.{minor}{letter}"
                                else:
                                    gcode = 'unknown'
                                fpath = os.path.join(so_dir, fn)
                                sz = os.path.getsize(fpath)
                                so_gencode_sizes[gcode] = so_gencode_sizes.get(gcode, 0) + sz
                    except Exception:
                        # fallback to list-only detection
                        codes = detect_gencodes(so_path)
                        for g in codes:
                            so_gencode_sizes[g] = so_gencode_sizes.get(g, 0) + size
                else:
                    # no cuobjdump: fallback to simple detection
                    codes = detect_gencodes(so_path)
                    for g in codes:
                        so_gencode_sizes[g] = so_gencode_sizes.get(g, 0) + size
                # record stats
                stats['so_files'].append((name, size, compressed_size, so_gencode_sizes))
                so_total += size
                for g, chunk in so_gencode_sizes.items():
                    stats['gencode_summary'][g] = stats['gencode_summary'].get(g, 0) + chunk
    stats['python_total'] = py_total
    stats['so_total'] = so_total
    stats['total_size'] = total_size
    return stats


def human_readable(size):  # pragma: no cover
    """Convert a size in bytes to a human-readable string."""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


# use rich to render tables
def print_report(stats):
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print("Warning: rich library not found; falling back to ASCII tables", file=sys.stderr)
        # fallback to ASCII formatting
        ascii_report = format_report(stats)
        print(ascii_report)
        return
    console = Console()

    # Summary table
    summary = Table(title="Wheel Size Summary")
    summary.add_column("Component", justify="left")
    summary.add_column("Size", justify="right")
    summary.add_column("% of total", justify="right")
    total = stats["total_size"]
    # Python code total
    py = stats["python_total"]
    summary.add_row(
        "Python code",
        human_readable(py),
        f"{py / total * 100:.1f}%"
    )
    # CUDA gencodes
    def _sort_key(item):
        key, _ = item
        try:
            return float(key)
        except:
            return float("inf")
    for gcode, sz in sorted(stats["gencode_summary"].items(), key=_sort_key):
        # numeric gencodes (with optional letter suffix) labeled as sm_XX
        if re.match(r"^\d+(?:\.\d+)?[a-z]?$", gcode, re.IGNORECASE):
            label = f"CUDA sm_{gcode}"
        else:
            label = f"CUDA {gcode}"
        summary.add_row(
            label,
            human_readable(sz),
            f"{sz / total * 100:.1f}%"
        )
    # Other files
    other_sz = stats["total_size"] - stats["python_total"] - stats["so_total"]
    if other_sz > 0:
        summary.add_row(
            "Other",
            human_readable(other_sz),
            f"{other_sz / total * 100:.1f}%"
        )
    summary.add_row(
        "Total",
        human_readable(total),
        "100.0%"
    )
    console.print(summary)

    # Details per shared object
    details = Table(title="Shared Object Breakdown")
    details.add_column("Shared object", justify="left")
    details.add_column("Total size", justify="right")
    details.add_column("Zipped size", justify="right")
    details.add_column("Gencode", justify="left")
    details.add_column("Chunk size", justify="right")
    details.add_column("% of SO", justify="right")
    # helper to sort gencodes by numeric part, then letter
    def _gencode_key(g):
        m = re.match(r"(\d+(?:\.\d+)?)([a-z]?)", g, re.IGNORECASE)
        if m:
            num = float(m.group(1))
            letter = m.group(2) or ''
            return (num, letter)
        return (float('inf'), g)
    # populate rows with row groups
    for path, size, compressed_size, gencodes_dict in stats["so_files"]:
        # if no gencodes detected
        if not gencodes_dict:
            details.add_row(path, human_readable(size), human_readable(compressed_size), "-", "-", "-")
            continue
        # sort gencodes
        items = sorted(gencodes_dict.items(), key=lambda kv: _gencode_key(kv[0]))
        for idx, (g, chunk_sz) in enumerate(items):
            # label numeric gencodes as sm_<code>
            if re.match(r"^\d+(?:\.\d+)?[a-z]?$$", g, re.IGNORECASE):
                label = f"sm_{g}"
            else:
                label = g
            # percent of this shared object
            pct = chunk_sz / size * 100 if size else 0
            pct_str = f"{pct:.1f}%"
            if idx == 0:
                details.add_row(path,
                                human_readable(size),
                                human_readable(compressed_size),
                                label,
                                human_readable(chunk_sz),
                                pct_str)
            else:
                details.add_row("", "", "", label, human_readable(chunk_sz), pct_str)
    console.print(details)


def format_report(stats):  # pragma: no cover
    """Format the analysis stats into a pretty text report."""
    lines = []
    lines.append("Summary:")
    # summary table rows
    total = stats['total_size']
    rows = []
    # Python code total
    pct_py = stats['python_total'] / total * 100 if total else 0
    rows.append((
        "Python code",
        human_readable(stats['python_total']),
        f"{pct_py:.1f}%"
    ))
    # CUDA gencode summary, sort numerically, 'other' last
    def _sort_key(item):
        key, _ = item
        try:
            return float(key)
        except Exception:
            return float('inf')
    for gcode, sz in sorted(stats['gencode_summary'].items(), key=_sort_key):
        # numeric gencodes (optional letter suffix) labeled as sm_<code>
        if re.match(r"^\d+(?:\.\d+)?[a-z]?$", gcode, re.IGNORECASE):
            label = f"CUDA sm_{gcode}"
        else:
            label = f"CUDA {gcode}"
        pct = sz / total * 100 if total else 0
        rows.append((
            label,
            human_readable(sz),
            f"{pct:.1f}%"
        ))
    # other files
    other_sz = stats['total_size'] - stats['python_total'] - stats['so_total']
    if other_sz > 0:
        pct_o = other_sz / total * 100 if total else 0
        rows.append((
            "Other",
            human_readable(other_sz),
            f"{pct_o:.1f}%"
        ))
    # total
    rows.append((
        "Total",
        human_readable(total),
        "100.0%"
    ))
    # column widths
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    w3 = max(len(r[2]) for r in rows)
    sep = f"+-{'-'*w1}-+-{'-'*w2}-+-{'-'*w3}-+"
    lines.append(sep)
    lines.append(f"| {'Component'.ljust(w1)} | {'Size'.ljust(w2)} | {'%'.ljust(w3)} |")
    lines.append(sep)
    for name, val, pct in rows:
        lines.append(f"| {name.ljust(w1)} | {val.rjust(w2)} | {pct.rjust(w3)} |")
    lines.append(sep)

    # details per shared object
    lines.append("\nDetails per shared object:")
    if stats['so_files']:
        # list each .so with its per-gencode chunk sizes
        def _gkey(g):
            m = re.match(r"(\d+(?:\.\d+)?)([a-z]?)", g, re.IGNORECASE)
            if m:
                return (float(m.group(1)), m.group(2) or '')
            return (float('inf'), g)
        for path, size, compressed_size, gencodes_dict in stats['so_files']:
            lines.append(f" {path} ({human_readable(size)}, zipped: {human_readable(compressed_size)}):")
            # sort and display each gencode
            for g, chunk in sorted(gencodes_dict.items(), key=lambda kv: _gkey(kv[0])):
                # label numeric codes
                if re.match(r"^\d+(?:\.\d+)?[a-z]?$$", g, re.IGNORECASE):
                    lbl = f"sm_{g}"
                else:
                    lbl = g
                # percent of this shared object chunk
                pct = chunk / size * 100 if size else 0
                lines.append(f"    {lbl:8s} {human_readable(chunk):>9s} ({pct:.1f}%)")
    else:
        lines.append(" <no shared objects>")

    return "\n".join(lines)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Generate wheel size report for vLLM wheel"
    )
    parser.add_argument(
        'source', help='Path or URL to .whl file'
    )
    args = parser.parse_args()
    source = args.source
    # download if URL
    if source.startswith(('http://', 'https://')):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.whl')
            print(f"Downloading {source} ...", file=sys.stderr)
            urllib.request.urlretrieve(source, tmp.name)
            print(f"Wheel downloaded to: {tmp.name}", file=sys.stderr)
            wheel_path = tmp.name
        except Exception as e:
            print(f"Error downloading wheel: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        wheel_path = source
    try:
        stats = analyze_wheel(wheel_path)
    except Exception as e:
        print(f"Error analyzing wheel: {e}", file=sys.stderr)
        sys.exit(1)
    # render with rich tables
    print_report(stats)


if __name__ == '__main__':  # pragma: no cover
    main()
