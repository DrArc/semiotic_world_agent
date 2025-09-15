#!/usr/bin/env python3
# track_usage.py — Trace used files/imports AND produce a requirements.txt of actually-used packages.
# Python 3.8+

import argparse, os, runpy, sys, sysconfig, site, trace, ast, pathlib, json, threading

# ---- compatibility shims for package metadata ----
try:
    import importlib.metadata as im  # stdlib in 3.8+
except Exception:
    im = None
try:
    import pkg_resources  # from setuptools; optional fallback
except Exception:
    pkg_resources = None

def norm(p):
    # robust path normalization (case-insensitive filesystems included)
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))

def guess_stdlib_dirs():
    """Return stdlib + site-packages dirs to ignore in coverage."""
    dirs = set()
    try:
        dirs.add(norm(sysconfig.get_path("stdlib")))
    except Exception:
        pass
    for getter in (site.getsitepackages, site.getusersitepackages):
        try:
            v = getter()
            if isinstance(v, (list, tuple)):
                for d in v: dirs.add(norm(d))
            elif isinstance(v, str):
                dirs.add(norm(v))
        except Exception:
            pass
    return {d for d in dirs if os.path.isdir(d)}

def build_import_graph(files, base_dir):
    """Very light static graph: executed file -> local files it imports (best-effort)."""
    edges = []
    base = pathlib.Path(base_dir)
    file_set = {norm(f) for f in files}
    for f in files:
        try:
            src = pathlib.Path(f).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=f)
        except Exception:
            continue
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names: imports.append(a.name)
            elif isinstance(n, ast.ImportFrom):
                if n.module: imports.append(n.module)
        for mod in imports:
            rel = pathlib.Path(*mod.split("."))
            cand1 = norm(str(base / (str(rel) + ".py")))
            cand2 = norm(str(base / rel / "__init__.py"))
            for cand in (cand1, cand2):
                if cand in file_set:
                    edges.append((norm(f), cand))
                    break
    return edges

# ----------- requirements helpers -----------
def build_top_level_to_dists_map():
    """Map top-level module name -> [distribution names]."""
    mapping = {}
    # Try stdlib API first (best)
    if im and hasattr(im, "packages_distributions"):
        try:
            pd = im.packages_distributions()
            # pd is already {top_level_module: [dist, ...]}
            return {k: list(v) for k, v in pd.items()}
        except Exception:
            pass
    # Fallback: pkg_resources top_level.txt
    if pkg_resources:
        for dist in pkg_resources.working_set:
            try:
                if dist.has_metadata("top_level.txt"):
                    for line in dist.get_metadata_lines("top_level.txt"):
                        tl = line.strip()
                        if tl:
                            mapping.setdefault(tl, []).append(dist.project_name)
            except Exception:
                # ignore broken metadata
                pass
        return mapping
    # Final fallback: empty (we'll try guessing)
    return {}

def get_dist_version(dist_name):
    """Return installed version for distribution or None."""
    if im:
        try:
            return im.version(dist_name)
        except Exception:
            pass
    if pkg_resources:
        try:
            return pkg_resources.get_distribution(dist_name).version
        except Exception:
            pass
    return None

def derive_requirements(external_py, external_ext):
    """
    From external imported modules, derive a set of (distribution, version)
    and a details list for diagnostics.
    """
    # Collect module names that came from outside project
    imported_modules = {item["module"] for item in external_py} | {item["module"] for item in external_ext}
    top_levels = {m.split(".")[0] for m in imported_modules if m}

    tl2dist = build_top_level_to_dists_map()
    resolved = {}
    details = []
    unresolved_top_levels = set()

    for tl in sorted(top_levels):
        dists = tl2dist.get(tl)
        if not dists:
            # Try guessing: sometimes the dist is the same as top-level
            ver = get_dist_version(tl)
            if ver:
                resolved[tl] = ver
                details.append({"top_level": tl, "distribution": tl, "version": ver, "mapping": "guessed"})
            else:
                unresolved_top_levels.add(tl)
                details.append({"top_level": tl, "distribution": None, "version": None, "mapping": "unresolved"})
            continue

        for dist in dists:
            ver = get_dist_version(dist)
            if ver:
                resolved[dist] = ver
                details.append({"top_level": tl, "distribution": dist, "version": ver, "mapping": "metadata"})
            else:
                details.append({"top_level": tl, "distribution": dist, "version": None, "mapping": "metadata_no_version"})

    req_lines = [f"{name}=={version}" for name, version in sorted(resolved.items(), key=lambda kv: kv[0].lower())]
    return req_lines, details, sorted(unresolved_top_levels)

# -------------- main tracer --------------
def main():
    ap = argparse.ArgumentParser(description="Track which scripts a Python program actually uses and emit requirements.")
    ap.add_argument("script", help="Path to the Python script to run")
    ap.add_argument("--project-root", default=None, help="Limit results to this root (default: script folder)")
    ap.add_argument("--dot", default=None, help="Write a Graphviz .dot of local import edges actually executed")
    ap.add_argument("--json-out", default=None, help="Write JSON report to this file")
    ap.add_argument("--requirements-out", default=None, help="Write used third-party packages (dist==version) to this file")
    ap.add_argument("--print-requirements", action="store_true", help="Print requirements to stdout")
    ap.add_argument("script_args", nargs=argparse.REMAINDER, help="Args for the target script (prefix with --)")
    args = ap.parse_args()

    target = norm(args.script)
    if not os.path.isfile(target):
        sys.exit(f"No such script: {target}")

    base_dir = norm(args.project_root or os.path.dirname(target))

    # Build ignore list for stdlib/site-packages, but NEVER ignore your project root
    ignore_all = guess_stdlib_dirs()
    ignore_dirs = {d for d in ignore_all if not norm(base_dir).startswith(d)}

    executed_files = set()
    imported_modnames = set()
    opened_py_files = set()
    spawned = []
    net_connects = []

    # Audit hook: imports, file opens, subprocess, network connects.
    def audit(event, a):
        try:
            if event == "import":
                if isinstance(a, tuple) and a:
                    imported_modnames.add(a[0])
            elif event == "open":
                path = a[0] if isinstance(a, tuple) and a else None
                if isinstance(path, str):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in (".py", ".pyw") and norm(path).startswith(base_dir):
                        opened_py_files.add(norm(path))
            elif event == "subprocess.Popen":
                if isinstance(a, tuple) and a:
                    cmd = a[0]
                    spawned.append({"cmd": list(map(str, cmd))} if isinstance(cmd, (list, tuple)) else {"cmd":[str(cmd)]})
            elif event == "socket.connect":
                if isinstance(a, tuple) and len(a) >= 2:
                    addr = a[1]
                    if isinstance(addr, tuple) and len(addr) >= 2:
                        host, port = addr[0], addr[1]
                        net_connects.append({"host": str(host), "port": int(port)})
        except Exception:
            # Never let the audit hook crash the run
            pass

    sys.addaudithook(audit)

    # Trace executed lines (include new threads)
    tracer = trace.Trace(count=True, trace=False, ignoredirs=list(ignore_dirs))
    threading.settrace(tracer.globaltrace)  # trace new threads too

    def runner():
        # Prepare argv for the target
        sys.argv = [target] + (args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args)
        runpy.run_path(target, run_name="__main__")

    try:
        tracer.runfunc(runner)
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"[target exited with code {e.code}]", file=sys.stderr)
    except Exception as e:
        print(f"[target raised {type(e).__name__}: {e}]", file=sys.stderr)

    # Collect executed files
    results = tracer.results()
    for (fname, _lineno), _count in results.counts.items():
        f = norm(fname)
        if f.startswith(base_dir) and f.endswith(".py"):
            executed_files.add(f)

    # Local imported files (under project_root)
    local_imported_files = set()
    for name in sorted(imported_modnames):
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None)
        if isinstance(f, str):
            f = norm(f)
            if f.endswith(".py") and f.startswith(base_dir):
                local_imported_files.add(f)

    # External imports (outside project_root)
    external_py = []
    external_ext = []
    imports_nofile = []  # builtins / namespace packages

    for name in sorted(imported_modnames):
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None)
        if not f:
            imports_nofile.append(name)
            continue
        f = norm(f)
        if f.startswith(base_dir):
            continue
        if f.endswith((".so", ".pyd", ".dll")):
            external_ext.append({"module": name, "path": f})
        else:
            external_py.append({"module": name, "path": f})

    # Derive requirements from external imports
    req_lines, req_details, unresolved = derive_requirements(external_py, external_ext)

    # Union of "executed" and "opened/imported" under the project
    used_scripts = sorted(set().union(executed_files, opened_py_files, local_imported_files))

    report = {
        "project_root": base_dir,
        "target": target,
        "used_scripts": [os.path.relpath(p, base_dir) for p in used_scripts],
        "executed_scripts": [os.path.relpath(p, base_dir) for p in sorted(executed_files)],
        "imported_local_scripts": [os.path.relpath(p, base_dir) for p in sorted(local_imported_files)],
        "opened_local_py_files": [os.path.relpath(p, base_dir) for p in sorted(opened_py_files)],
        "external_imports_py": external_py,
        "external_imports_ext": external_ext,
        "imports_without_file": imports_nofile,
        "requirements": req_lines,
        "requirements_details": req_details,
        "requirements_unresolved_top_levels": unresolved,
        "subprocesses": spawned,
        "network_connections": net_connects,
    }

    out = json.dumps(report, indent=2)
    print(out)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"[wrote {args.json_out}]", file=sys.stderr)

    if args.print_requirements:
        for line in req_lines:
            print(line)

    if args.requirements_out:
        with open(args.requirements_out, "w", encoding="utf-8") as fh:
            for line in req_lines:
                fh.write(line + "\n")
            if unresolved:
                fh.write("\n# Unresolved top-level modules (no matching dist found):\n")
                for tl in unresolved:
                    fh.write(f"# - {tl}\n")
        print(f"[wrote {args.requirements_out}]", file=sys.stderr)

    # Optional: DOT graph of import edges among executed local files
    if args.dot:
        edges = build_import_graph(executed_files, base_dir)
        with open(args.dot, "w", encoding="utf-8") as fh:
            fh.write("digraph deps {\n")
            fh.write('  graph [rankdir=LR]; node [shape=box, fontsize=10];\n')
            def label(p): return os.path.relpath(p, base_dir).replace("\\", "/")
            nodes = {n for pair in edges for n in pair}
            for n in sorted(nodes):
                fh.write(f'  "{label(n)}";\n')
            for a, b in edges:
                fh.write(f'  "{label(a)}" -> "{label(b)}";\n')
            fh.write("}\n")
        print(f'Wrote Graphviz DOT: {args.dot}', file=sys.stderr)

if __name__ == "__main__":
    main()