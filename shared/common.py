import subprocess, threading
from pathlib import Path
from datetime import datetime

def ts():
    # add microseconds so two clicks in the same second don't overwrite the file
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_subprocess(python_exe: Path, script: Path, args):
    # keep this for simple, non-streaming calls
    cmd = [str(python_exe), str(script)] + [str(a) for a in args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)

def run_subprocess_stream(python_exe: Path, script: Path, args, on_stdout=None, on_stderr=None):
    """Stream stdout/stderr line-by-line to callbacks so the UI can show progress."""
    cmd = [str(python_exe), str(script)] + [str(a) for a in args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, universal_newlines=True)

    def pump(stream, cb):
        if stream is None: return
        for line in iter(stream.readline, ''):
            if cb:
                cb(line.rstrip())
        stream.close()

    t1 = threading.Thread(target=pump, args=(proc.stdout, on_stdout), daemon=True)
    t2 = threading.Thread(target=pump, args=(proc.stderr, on_stderr), daemon=True)
    t1.start(); t2.start()

    code = proc.wait()
    t1.join(); t2.join()
    if code != 0:
        raise RuntimeError(f"{script.name} failed with exit code {code}")
