"""Minify static CSS and JS assets in place."""
import os

from cssmin import cssmin
from jsmin import jsmin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSETS = [
    os.path.join(BASE_DIR, "static", "css", "style.css"),
    os.path.join(BASE_DIR, "static", "js", "bundle.js"),
]


def minify_file(path: str, minifier) -> None:
    before = os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    minified = minifier(original)
    with open(path, "w", encoding="utf-8") as f:
        f.write(minified)
    after = os.path.getsize(path)
    print(f"{path}: {before} -> {after} bytes ({100 - after * 100 // before}% smaller)")


def main() -> None:
    minify_file(ASSETS[0], cssmin)
    minify_file(ASSETS[1], jsmin)


if __name__ == "__main__":
    main()
