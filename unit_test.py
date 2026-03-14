"""
Unit Test Automator — AST-based chunking + OpenAI Responses API + pytest
=========================================================================
Usage:
    python unit_test_automator.py <path_to_your_file.py> [options]

Options:
    --output-dir   Where to write generated test file (default: ./generated_tests)
    --model        OpenAI model to use (default: gpt-4o-mini)
    --run          Automatically run pytest after generation (default: True)
    --verbose      Show full pytest output

Requirements:
    pip install openai pytest
    export OPENAI_API_KEY=sk-...
"""

import ast
import sys
import os
import re
import subprocess
import textwrap
import argparse
from pathlib import Path
from typing import Optional
from openai import OpenAI


# ─────────────────────────────────────────────────────────────────────────────
# 1. AST PARSER — Extract individual functions and classes from source
# ─────────────────────────────────────────────────────────────────────────────

class ASTChunker:
    """
    Parses a .py file and extracts each function / class as an independent
    'chunk' with its source code, docstring, and any top-level imports.
    This lets us send one small, focused unit to the LLM per call.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, "r", encoding="utf-8") as f:
            self.source = f.read()
        self.lines = self.source.splitlines()
        self.tree = ast.parse(self.source, filename=filepath)

    # ── collect all top-level imports ──────────────────────────────────────
    def _get_imports(self) -> str:
        import_lines = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # only top-level (direct children of module)
                if node in ast.walk(self.tree) and node.col_offset == 0:
                    import_lines.append(ast.get_source_segment(self.source, node))
        # deduplicate, preserve order
        seen = set()
        deduped = []
        for line in import_lines:
            if line and line not in seen:
                seen.add(line)
                deduped.append(line)
        return "\n".join(deduped)

    # ── extract source segment safely ───────────────────────────────────────
    def _node_source(self, node: ast.AST) -> str:
        segment = ast.get_source_segment(self.source, node)
        return segment or ""

    # ── build chunks ────────────────────────────────────────────────────────
    def get_chunks(self) -> list[dict]:
        """
        Returns a list of dicts, one per function/class:
        {
            "name":        str,
            "type":        "function" | "class",
            "source":      str,          # just this unit
            "imports":     str,          # all file-level imports
            "class_name":  str | None,   # set for methods
            "docstring":   str | None,
            "lineno":      int,
        }
        """
        imports = self._get_imports()
        chunks = []

        for node in ast.walk(self.tree):
            # ── standalone top-level functions ──────────────────────────────
            if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                chunks.append({
                    "name": node.name,
                    "type": "function",
                    "source": self._node_source(node),
                    "imports": imports,
                    "class_name": None,
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                })

            # ── classes (include full class so methods have context) ─────────
            elif isinstance(node, ast.ClassDef) and node.col_offset == 0:
                class_source = self._node_source(node)
                # add one chunk for the entire class
                chunks.append({
                    "name": node.name,
                    "type": "class",
                    "source": class_source,
                    "imports": imports,
                    "class_name": node.name,
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                })

        # sort by line number so the generated test file follows source order
        chunks.sort(key=lambda c: c["lineno"])
        return chunks

    def summary(self) -> str:
        chunks = self.get_chunks()
        lines = [f"  [{c['type']:8s}] {c['name']} (line {c['lineno']})" for c in chunks]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST GENERATOR — One LLM call per chunk via OpenAI Responses API
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python test engineer. 
Your job is to write pytest unit tests for the given function or class.

Rules:
- Use pytest (no unittest)
- Cover: happy path, edge cases, boundary values, error/exception cases
- Use parametrize for multiple similar inputs
- Mock external dependencies (file I/O, network, DB) using unittest.mock
- Do NOT import the function yourself — the caller handles imports
- Return ONLY raw Python code, no markdown, no backticks, no explanation
- Each test function must start with test_
- Add a brief one-line comment above each test explaining what it tests
- If you cannot determine behavior, add a TODO comment in the test
"""

def build_user_prompt(chunk: dict) -> str:
    unit_type = chunk["type"]
    name = chunk["name"]
    source = chunk["source"]
    imports = chunk["imports"]
    docstring = chunk.get("docstring") or ""

    prompt = f"""Generate pytest unit tests for this Python {unit_type}.

=== FILE IMPORTS (for context) ===
{imports if imports else "# (no imports)"}

=== {unit_type.upper()} SOURCE ===
{source}
"""
    if docstring:
        prompt += f"\n=== DOCSTRING ===\n{docstring}\n"

    prompt += f"""
=== YOUR TASK ===
Write comprehensive pytest tests for `{name}`.
- Assume the function/class is importable as: from <module> import {name}
- Return only Python test code — no markdown, no preamble, no explanation.
"""
    return prompt


class TestGenerator:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate_for_chunk(self, chunk: dict) -> str:
        """Call OpenAI Responses API for one chunk, return raw test code."""
        user_prompt = build_user_prompt(chunk)
        try:
            # ── OpenAI Responses API ───────────────────────────────────────
            response = self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
            )
            raw = response.output_text.strip()

            # strip any accidental markdown fences
            raw = re.sub(r"^```python\s*", "", raw)
            raw = re.sub(r"^```\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            return raw.strip()

        except Exception as e:
            # return a placeholder test so the file still runs
            return (
                f"# ⚠️  Could not generate tests for `{chunk['name']}`: {e}\n"
                f"def test_{chunk['name']}_placeholder():\n"
                f"    pass  # TODO: generation failed\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEST MERGER — Combine all generated chunks into one valid test file
# ─────────────────────────────────────────────────────────────────────────────

def merge_test_file(
    source_filepath: str,
    chunks: list[dict],
    test_bodies: list[str],
    output_path: str,
) -> str:
    """Build a single test_*.py file from all per-chunk test bodies."""

    module_name = Path(source_filepath).stem
    import_line = f"from {module_name} import *"

    header = textwrap.dedent(f"""\
        # =============================================================
        # Auto-generated tests for: {Path(source_filepath).name}
        # Generated by unit_test_automator.py
        # =============================================================
        import pytest
        from unittest.mock import patch, MagicMock, mock_open
        {import_line}

    """)

    sections = []
    for chunk, body in zip(chunks, test_bodies):
        sep = f"# {'─' * 60}\n# Tests for {chunk['type']}: {chunk['name']}\n# {'─' * 60}\n"
        sections.append(sep + body + "\n")

    full_content = header + "\n\n".join(sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    return full_content


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYTEST RUNNER — Run tests and return structured results
# ─────────────────────────────────────────────────────────────────────────────

def run_pytest(test_file: str, source_dir: str, verbose: bool = False) -> dict:
    """
    Run pytest on the generated test file.
    Returns dict with: passed, failed, errors, output, returncode
    """
    flags = ["-v"] if verbose else ["-v", "--tb=short"]
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        *flags,
        "--color=no",           # clean output for parsing
        f"--rootdir={source_dir}",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=source_dir,
    )
    output = result.stdout + result.stderr

    # parse summary line: "X passed, Y failed, Z error"
    passed = int(re.search(r"(\d+) passed", output).group(1)) if "passed" in output else 0
    failed = int(re.search(r"(\d+) failed", output).group(1)) if "failed" in output else 0
    errors = int(re.search(r"(\d+) error",  output).group(1)) if "error"  in output else 0

    return {
        "passed":     passed,
        "failed":     failed,
        "errors":     errors,
        "total":      passed + failed + errors,
        "output":     output,
        "returncode": result.returncode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ORCHESTRATOR — Wire everything together
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    filepath: str,
    output_dir: str = "./generated_tests",
    model: str = "gpt-4o-mini",
    run_tests: bool = True,
    verbose: bool = False,
) -> None:

    filepath = os.path.abspath(filepath)
    source_dir = str(Path(filepath).parent)
    module_name = Path(filepath).stem
    os.makedirs(output_dir, exist_ok=True)
    output_test_path = os.path.join(output_dir, f"test_{module_name}.py")

    print(f"\n{'='*60}")
    print(f"  Unit Test Automator")
    print(f"  Source : {filepath}")
    print(f"  Output : {output_test_path}")
    print(f"  Model  : {model}")
    print(f"{'='*60}\n")

    # ── Step 1: Parse with AST ─────────────────────────────────────────────
    print("[ 1/4 ] Parsing source with AST...")
    chunker = ASTChunker(filepath)
    chunks = chunker.get_chunks()

    if not chunks:
        print("  ⚠️  No functions or classes found in the file.")
        return

    print(f"  Found {len(chunks)} unit(s) to test:")
    print(chunker.summary())

    # ── Step 2: Generate tests per chunk ──────────────────────────────────
    print(f"\n[ 2/4 ] Generating tests via {model}...")
    generator = TestGenerator(model=model)
    test_bodies = []

    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {chunk['type']}: {chunk['name']}...", end=" ", flush=True)
        body = generator.generate_for_chunk(chunk)
        test_bodies.append(body)
        print("✓")

    # ── Step 3: Merge into single test file ───────────────────────────────
    print(f"\n[ 3/4 ] Merging test file → {output_test_path}")
    merge_test_file(filepath, chunks, test_bodies, output_test_path)
    print(f"  ✓ Written ({sum(1 for l in open(output_test_path))} lines)")

    # ── Step 4: Run pytest ─────────────────────────────────────────────────
    if run_tests:
        print(f"\n[ 4/4 ] Running pytest...\n")
        # add source dir to PYTHONPATH so imports work
        env = os.environ.copy()
        env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")

        results = run_pytest(
            test_file=output_test_path,
            source_dir=source_dir,
            verbose=verbose,
        )

        if verbose:
            print(results["output"])
        else:
            # print last 30 lines (summary)
            tail = "\n".join(results["output"].strip().splitlines()[-30:])
            print(tail)

        print(f"\n{'─'*60}")
        print(f"  Results  :  {results['passed']} passed  "
              f"{results['failed']} failed  {results['errors']} errors")
        print(f"  Test file:  {output_test_path}")
        print(f"{'─'*60}\n")
    else:
        print(f"\n  Test file written. Run manually:\n  pytest {output_test_path} -v\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate and run pytest unit tests for a Python file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python unit_test_automator.py my_module.py
          python unit_test_automator.py utils.py --output-dir ./tests --verbose
          python unit_test_automator.py services.py --model gpt-4o --no-run
        """)
    )
    parser.add_argument(
        "filepath",
        help="Path to the .py file you want to test"
    )
    parser.add_argument(
        "--output-dir", default="./generated_tests",
        help="Directory for generated test file (default: ./generated_tests)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="Only generate test file, don't run pytest"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full pytest output"
    )
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: file not found: {args.filepath}")
        sys.exit(1)
    if not args.filepath.endswith(".py"):
        print(f"Warning: expected a .py file, got: {args.filepath}")

    run_pipeline(
        filepath=args.filepath,
        output_dir=args.output_dir,
        model=args.model,
        run_tests=not args.no_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()