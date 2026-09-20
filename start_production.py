#!/usr/bin/env python3

import subprocess
import signal
import sys
import time
import os

gunicorn_process = None
llama_process = None

LLAMA_MODEL = "/media/novadev/storage/daylivotion/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
LLAMA_PORT = "8981"


def start_llama_server():
    global llama_process

    print("Starting llama server...")

    llama_process = subprocess.Popen(
        [
            "llama",
            "serve",
            "-m",
            LLAMA_MODEL,
            "--port",
            LLAMA_PORT,
            "--host",
            "127.0.0.1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    print("Waiting for llama server to start...")

    while True:
        if llama_process.poll() is not None:
            print("Llama server died unexpectedly.")
            return False

        line = llama_process.stderr.readline()

        if line:
            print(f"[llama] {line.rstrip()}")

            if f"listening on http://127.0.0.1:{LLAMA_PORT}" in line:
                print("Llama server is ready!")
                return True

    return False


def build_tailwind():
    print("Building Tailwind CSS...")
    try:
        # First install npm dependencies if needed
        print("Installing npm dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd="/home/novadev/server/daylivotion",
            check=True,
            capture_output=True,
            text=True
        )
        
        # Run npx from the project directory where package.json is located
        result = subprocess.run(
            [
                "npx",
                "tailwindcss",
                "-i",
                "./static/css/input.css",
                "-o",
                "./static/css/output.css",
                "--minify"
            ],
            cwd="/home/novadev/server/daylivotion",
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": os.environ.get("PATH", "")}
        )
        print("Tailwind CSS built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Tailwind CSS: {e}")
        print(f"Error output: {e.stderr}")
        print("WARNING: Will use CDN for Tailwind CSS instead")
        return False
    except FileNotFoundError:
        print("npm/npx not found. Will use CDN for Tailwind CSS instead")
        return False


def start_gunicorn():
    global gunicorn_process

    print("Starting Gunicorn server...")
    gunicorn_process = subprocess.Popen(
        [
            "uv",
            "run",
            "--directory",
            "/media/novadev/storage/daylivotion/kokoro",
            "gunicorn",
            "--worker-class",
            "gevent",
            "-w",
            "1",
            "--timeout",
            "300",
            "--bind",
            "0.0.0.0:5100",
            "--chdir",
            "/home/novadev/server/daylivotion",
            "app:app"
        ]
    )

    return True


def stop_process(process, name):
    if process is None:
        return

    if process.poll() is None:
        print(f"Stopping {name}...")

        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"Force killing {name}...")
            process.kill()
            process.wait()


def cleanup_and_exit(signum=None, frame=None):
    global llama_process, gunicorn_process

    print("")
    print("Cleaning up before exit...")

    # Stop Gunicorn first
    stop_process(gunicorn_process, "Gunicorn")

    # Then stop Llama
    stop_process(llama_process, "llama server")

    print("Exit complete")
    sys.exit(0)


# Setup signal handlers
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)


if __name__ == "__main__":

    print("=" * 60)
    print("Starting Daylivotion production server")
    print("=" * 60)

    # Start Llama
    if not start_llama_server():
        print("Failed to start llama server.")
        cleanup_and_exit()
        sys.exit(1)

    # Build Tailwind CSS (optional)
    build_tailwind()

    # Start Gunicorn
    if not start_gunicorn():
        print("Failed to start Gunicorn.")
        cleanup_and_exit()
        sys.exit(1)

    print("")
    print("=" * 60)
    print("All servers started successfully!")
    print("=" * 60)
    print("Llama server : http://127.0.0.1:8981")
    print("Gunicorn     : http://0.0.0.0:5100")
    print("=" * 60)

    # Wait for both processes
    try:
        while True:

            if llama_process.poll() is not None:
                print("Llama server died unexpectedly!")
                cleanup_and_exit()
                sys.exit(1)

            if gunicorn_process.poll() is not None:
                print("Gunicorn died unexpectedly!")
                cleanup_and_exit()
                sys.exit(1)

            time.sleep(1)

    except KeyboardInterrupt:
        cleanup_and_exit()