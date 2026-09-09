import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

HELP = """usage: kavi <command> [arguments]

commands:
  train [options]
      Run train.py with the supplied options.

  infer <prompt> [options]
      Run train.py in inference mode, passing the prompt through stdin.

  render <prompt> [options]
      Run render.py with the supplied prompt.

examples:
  uv run kavi train --epochs 4
  uv run kavi infer "The king entered the forest"
  uv run kavi render "The king entered the forest"
"""


def script_path(name):
    path = ROOT / name
    if path.exists():
        return path

    path = Path.cwd() / name
    if path.exists():
        return path

    raise FileNotFoundError(f"Could not locate {name}")


def run_script(name, arguments, stdin=None):
    command = [sys.executable, str(script_path(name)), *arguments]
    result = subprocess.run(command, input=stdin, text=True)
    return result.returncode


def require_prompt(command, arguments):
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(f'usage: kavi {command} <prompt> [options]')
        return None, None

    return arguments[0], arguments[1:]


def main():
    arguments = sys.argv[1:]

    if not arguments or arguments[0] in {"-h", "--help"}:
        print(HELP)
        return

    command, *arguments = arguments

    if command == "train":
        raise SystemExit(run_script("train.py", arguments))

    if command == "infer":
        prompt, arguments = require_prompt(command, arguments)
        if prompt is None:
            return
        raise SystemExit(
            run_script("train.py", ["--infer-only", *arguments], stdin=prompt)
        )

    if command == "render":
        prompt, arguments = require_prompt(command, arguments)
        if prompt is None:
            return
        raise SystemExit(
            run_script("render.py", ["--prompt", prompt, *arguments])
        )

    print(f"Unknown command: {command}", file=sys.stderr)
    print(HELP, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
