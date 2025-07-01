import subprocess
import time
import os
import signal
import pytest

@pytest.mark.timeout(10)
def test_main_script_runs_without_crashing_for_5_seconds():
    script_path = os.path.join(os.path.dirname(__file__), "../main.py")

    proc = subprocess.Popen(
        ["python", script_path, "--spoof_k2eg_data", "--disable_file_logging"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    try:
        time.sleep(5)
        if proc.poll() is None:
            # send interrupt to shut down cleanly
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)

        stdout, stderr = proc.communicate(timeout=5)

        stdout_str = stdout.decode()
        stderr_str = stderr.decode()

        print("\n--- STDOUT ---")
        print(stdout_str)
        print("\n--- STDERR ---")
        print(stderr_str)

        print("!!! proc.returncode: ", proc.returncode)
        assert proc.returncode == 0, f"Process exited with code {proc.returncode}\nStderr:\n{stderr_str}"

    finally:
        # Cleanup if needed
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
