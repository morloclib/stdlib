#!/usr/bin/env python3
"""
libman.py - Manage the morloc standard library.

Replaces lib/stdlib/run-tests.sh with a richer Python tool that
understands "families" (groups of related modules: e.g. root + root-cpp +
root-py + root-r) and exposes them as the unit of operation for version
bumps, parallel git commits/pushes, and the freeze command that
regenerates the `stdlib` umbrella's morloc-deps pin list.

Stdlib-only: no external dependencies. Python 3.8+.

Subcommands:
    groups                List discovered families
    status                Per-module git status
    test                  Run morloc test suite per module
    diff                  Per-member git diff for a family
    add                   git add across a family
    commit                git commit across a family
    push                  git push across a family
    bump                  Increment version field across a family
    freeze                Write current HEAD hashes into stdlib/package.yaml

Run `./libman.py <subcommand> --help` for per-command flags.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _help_width() -> int:
    """Standard Unix help width: $COLUMNS if set, else terminal width,
    else 80. Argparse uses the same fallback chain via
    shutil.get_terminal_size(), but RawDescriptionHelpFormatter does not
    wrap descriptions, so we apply the width ourselves. We subtract 2
    for the same right-margin breathing room argparse uses."""
    return max(40, shutil.get_terminal_size((80, 24)).columns - 2)


def _D(text: str) -> str:
    """Normalize a help description and wrap it to the help width.

    Collapses internal whitespace (so multi-line Python source is joined
    into a single paragraph), then re-wraps. Because the wrapping is
    done at parser-construction time rather than format-time, it
    snapshots the width when the script is invoked -- which is exactly
    the same moment argparse itself reads COLUMNS, so the two stay in
    sync."""
    return textwrap.fill(" ".join(text.split()), width=_help_width())


# ---------------------------------------------------------------------------
# Constants and color codes
# ---------------------------------------------------------------------------

LANG_SUFFIXES = ("-cpp", "-py", "-r", "-julia")
DEFAULT_BRANCHES = ("master", "main")
UMBRELLA_NAME = "stdlib"
SKIP_DIRS = frozenset({"pools", ".claude", "__pycache__"})

# Modules that should be excluded from family-level operations even though
# they are valid stdlib members (the umbrella + dot-files etc.).
NON_FAMILY_NAMES = frozenset({UMBRELLA_NAME})

# ANSI colors. Suppressed when stdout is not a TTY.
_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _TTY else ""


RED = _c("\033[0;31m")
GREEN = _c("\033[0;32m")
YELLOW = _c("\033[0;33m")
BLUE = _c("\033[0;34m")
CYAN = _c("\033[0;36m")
BOLD = _c("\033[1m")
DIM = _c("\033[2m")
NC = _c("\033[0m")


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Module:
    """A single stdlib module on disk."""

    name: str
    path: Path
    family: str  # the stem (e.g. "root" for root, root-cpp, root-py, root-r)
    role: str    # "parent" | "lang" | "singleton" | "umbrella"

    @property
    def package_yaml(self) -> Path:
        return self.path / "package.yaml"

    @property
    def is_git(self) -> bool:
        return (self.path / ".git").exists()


@dataclass
class Family:
    """A group of related modules sharing a stem."""

    stem: str
    members: List[Module] = field(default_factory=list)

    def add(self, m: Module) -> None:
        self.members.append(m)

    @property
    def is_singleton(self) -> bool:
        return len(self.members) == 1

    def sorted(self) -> List[Module]:
        # Parent first (if any), then language impls alphabetically.
        return sorted(
            self.members,
            key=lambda m: (m.role != "parent", m.name),
        )


def stdlib_root() -> Path:
    """Return the directory containing this script (lib/stdlib/)."""
    return Path(__file__).resolve().parent


def family_stem(name: str) -> str:
    """Strip the language suffix (if any) to find the family stem."""
    for suffix in LANG_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def role_of(name: str, family: str, family_size: int) -> str:
    if name == UMBRELLA_NAME:
        return "umbrella"
    if any(name.endswith(s) for s in LANG_SUFFIXES):
        return "lang"
    if name == family and family_size > 1:
        return "parent"
    return "singleton"


def discover() -> Tuple[Dict[str, Family], Dict[str, Module]]:
    """Scan lib/stdlib/* and group into families.

    Returns (families_by_stem, modules_by_name).
    """
    root = stdlib_root()
    modules_by_name: Dict[str, Module] = {}
    stems: Dict[str, set] = {}  # stem -> {names}

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_DIRS:
            continue
        if entry.name.startswith("."):
            continue
        if not (entry / "package.yaml").exists():
            continue
        stem = family_stem(entry.name)
        stems.setdefault(stem, set()).add(entry.name)

    families: Dict[str, Family] = {}
    for stem, names in stems.items():
        fam = Family(stem=stem)
        size = len(names)
        for name in sorted(names):
            path = root / name
            role = role_of(name, stem, size)
            m = Module(name=name, path=path, family=stem, role=role)
            modules_by_name[name] = m
            fam.add(m)
        families[stem] = fam

    return families, modules_by_name


# ---------------------------------------------------------------------------
# Group selector
# ---------------------------------------------------------------------------


def select_modules(
    families: Dict[str, Family],
    modules: Dict[str, Module],
    *,
    group: Optional[str] = None,
    all_: bool = False,
    lang_only: bool = False,
    match: Optional[str] = None,
    include_umbrella: bool = False,
) -> List[Module]:
    """Resolve a CLI selector into a list of modules.

    The umbrella (`stdlib`) is excluded from group ops by default.
    """
    if sum(bool(x) for x in (group, all_, match)) > 1:
        die("group selector: pass at most one of GROUP, --all, --match")

    selected: List[Module] = []

    if all_ or lang_only or match:
        for name, m in sorted(modules.items()):
            if m.role == "umbrella" and not include_umbrella:
                continue
            if lang_only and m.role != "lang":
                continue
            if match is not None and not re.search(match, name):
                continue
            selected.append(m)
    elif group is not None:
        if group in families:
            selected = list(families[group].sorted())
        elif group in modules:
            # Allow targeting a single module by name.
            selected = [modules[group]]
        else:
            die(f"unknown group or module: {group!r}")
    else:
        die("group selector required (pass GROUP, --all, --lang-only, or --match)")

    if not selected:
        die("selector resolved to zero modules")
    return selected


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    capture: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess, returning the CompletedProcess."""
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        capture_output=capture,
        text=True,
        check=check,
    )


def git(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return run(["git", *args], cwd=cwd)


def die(msg: str, code: int = 1) -> None:
    sys.stdout.flush()
    print(f"{RED}error:{NC} {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Git inspection helpers (read-only)
# ---------------------------------------------------------------------------


def git_head_hash(m: Module) -> Optional[str]:
    if not m.is_git:
        return None
    r = git(["rev-parse", "HEAD"], m.path)
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def git_branch(m: Module) -> Optional[str]:
    """Current branch name, or None if detached."""
    r = git(["symbolic-ref", "--short", "HEAD"], m.path)
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def git_has_upstream(m: Module) -> bool:
    r = git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], m.path)
    return r.returncode == 0


def git_has_unstaged(m: Module) -> bool:
    r = git(["diff", "--quiet"], m.path)
    return r.returncode != 0


def git_has_staged(m: Module) -> bool:
    r = git(["diff", "--cached", "--quiet"], m.path)
    return r.returncode != 0


def git_untracked(m: Module) -> List[str]:
    r = git(["ls-files", "--others", "--exclude-standard"], m.path)
    if r.returncode != 0:
        return []
    return [ln for ln in r.stdout.splitlines() if ln]


def git_ahead_behind(m: Module) -> Optional[Tuple[int, int]]:
    """Returns (ahead, behind) counts vs upstream, or None if no upstream."""
    if not git_has_upstream(m):
        return None
    a = git(["rev-list", "--count", "@{upstream}..HEAD"], m.path)
    b = git(["rev-list", "--count", "HEAD..@{upstream}"], m.path)
    if a.returncode != 0 or b.returncode != 0:
        return None
    try:
        return int(a.stdout.strip()), int(b.stdout.strip())
    except ValueError:
        return None


def git_has_commit(m: Module) -> bool:
    r = git(["rev-parse", "--verify", "HEAD"], m.path)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


@dataclass
class CheckProfile:
    """Which checks apply to which subcommand."""

    needs_git: bool = False
    needs_commit: bool = False
    needs_clean: bool = False  # subject to --allow-dirty
    needs_branch: bool = False  # not detached
    needs_upstream: bool = False  # subject to --set-upstream
    warns_default_branch: bool = False


PROFILES: Dict[str, CheckProfile] = {
    "bump":   CheckProfile(needs_clean=True),
    "add":    CheckProfile(needs_git=True),
    "commit": CheckProfile(
        needs_git=True, needs_branch=True, warns_default_branch=True
    ),
    "push": CheckProfile(
        needs_git=True,
        needs_commit=True,
        needs_branch=True,
        needs_upstream=True,
        warns_default_branch=True,
    ),
    "freeze": CheckProfile(needs_git=True, needs_commit=True, needs_clean=True),
}


@dataclass
class PreflightFinding:
    module: str
    check: str
    detail: str = ""


def preflight(
    modules: List[Module],
    profile: CheckProfile,
    *,
    allow_dirty: bool = False,
    set_upstream: bool = False,
) -> Tuple[List[PreflightFinding], List[PreflightFinding]]:
    """Run all checks against all modules. Return (errors, warnings)."""
    errors: List[PreflightFinding] = []
    warnings: List[PreflightFinding] = []

    for m in modules:
        if not m.path.exists():
            errors.append(PreflightFinding(m.name, "module dir missing"))
            continue
        if not m.package_yaml.exists():
            errors.append(PreflightFinding(m.name, "package.yaml missing"))
            continue
        if profile.needs_git and not m.is_git:
            errors.append(PreflightFinding(m.name, "no .git"))
            continue
        if profile.needs_commit and not git_has_commit(m):
            errors.append(PreflightFinding(m.name, "no commits on HEAD"))
            continue
        if profile.needs_clean and not allow_dirty and m.is_git:
            if git_has_unstaged(m) or git_has_staged(m):
                errors.append(
                    PreflightFinding(m.name, "working tree not clean",
                                     "(pass --allow-dirty to override)")
                )
        if profile.needs_branch:
            br = git_branch(m)
            if br is None:
                errors.append(PreflightFinding(m.name, "detached HEAD"))
                continue
            if profile.warns_default_branch and br not in DEFAULT_BRANCHES:
                warnings.append(
                    PreflightFinding(m.name, "non-default branch", f"({br})")
                )
        if profile.needs_upstream and not set_upstream:
            if not git_has_upstream(m):
                errors.append(
                    PreflightFinding(m.name, "no tracked upstream",
                                     "(pass --set-upstream to push as new branch)")
                )

    return errors, warnings


def report_preflight(
    errors: List[PreflightFinding], warnings: List[PreflightFinding]
) -> None:
    if warnings:
        print(f"{YELLOW}warnings:{NC}")
        by_check: Dict[str, List[PreflightFinding]] = {}
        for w in warnings:
            by_check.setdefault(w.check, []).append(w)
        for check, items in by_check.items():
            print(f"  {check}:")
            for it in items:
                detail = f" {it.detail}" if it.detail else ""
                print(f"    - {it.module}{detail}")

    if errors:
        print(f"{RED}error:{NC} pre-flight checks failed for "
              f"{len({e.module for e in errors})} module(s):")
        by_check: Dict[str, List[PreflightFinding]] = {}
        for e in errors:
            by_check.setdefault(e.check, []).append(e)
        for check, items in by_check.items():
            print(f"  {check}:")
            for it in items:
                detail = f"  {it.detail}" if it.detail else ""
                print(f"    - {it.module}{detail}")
        print("no operations performed.")


# ---------------------------------------------------------------------------
# YAML helpers (line-oriented, no pyyaml)
# ---------------------------------------------------------------------------


_VERSION_RE = re.compile(r"^(version:\s*)(\S+)\s*$", re.MULTILINE)
_DEPS_BLOCK_RE = re.compile(
    r"^morloc-deps:[ \t]*(?:\n(?:[ \t]+.*\n?)*)?",
    re.MULTILINE,
)


def get_version(text: str) -> Optional[str]:
    m = _VERSION_RE.search(text)
    return m.group(2) if m else None


def set_version(text: str, version: str) -> str:
    m = _VERSION_RE.search(text)
    if not m:
        raise ValueError("no `version:` field in package.yaml")
    return text[: m.start(2)] + version + text[m.end(2):]


def set_morloc_deps(text: str, entries: List[Tuple[str, str]]) -> str:
    """Replace (or append) the morloc-deps block."""
    if entries:
        block_lines = ["morloc-deps:"]
        for n, h in entries:
            block_lines.append(f"  - name: {n}")
            block_lines.append(f"    git-hash: {h}")
        block = "\n".join(block_lines) + "\n"
    else:
        block = "morloc-deps: []\n"

    if _DEPS_BLOCK_RE.search(text):
        return _DEPS_BLOCK_RE.sub(block, text, count=1)

    if not text.endswith("\n"):
        text += "\n"
    return text + block


def bump_version_string(version: str, level: str) -> str:
    """Apply a SemVer-style bump. Tolerates a leading 'v'."""
    prefix = "v" if version.startswith("v") else ""
    core = version[1:] if prefix else version
    parts = core.split(".")
    while len(parts) < 3:
        parts.append("0")
    if len(parts) > 3:
        die(f"version {version!r} has more than 3 components; refusing to bump")
    try:
        major, minor, patch = (int(p) for p in parts[:3])
    except ValueError:
        die(f"version {version!r} is not numeric SemVer; refusing to bump")
    if level == "major":
        major, minor, patch = major + 1, 0, 0
    elif level == "minor":
        minor, patch = minor + 1, 0
    elif level == "patch":
        patch += 1
    else:
        die(f"bad bump level {level!r} (expected major|minor|patch)")
    return f"{prefix}{major}.{minor}.{patch}"


# ---------------------------------------------------------------------------
# Subcommand: groups
# ---------------------------------------------------------------------------


def cmd_groups(args: argparse.Namespace) -> int:
    families, _ = discover()
    print(f"{BOLD}{BLUE}stdlib families ({len(families)} total){NC}\n")

    def show(fam: Family) -> None:
        members = fam.sorted()
        roles = ", ".join(f"{m.name}{DIM}({m.role[0]}){NC}" for m in members)
        marker = ""
        if fam.is_singleton:
            marker = f"  {DIM}[singleton]{NC}"
        print(f"  {BOLD}{fam.stem}{NC}{marker}")
        print(f"    {roles}")

    visible = sorted(families.values(), key=lambda f: f.stem)
    if args.lang_only:
        for fam in visible:
            members = [m for m in fam.sorted() if m.role == "lang"]
            if not members:
                continue
            print(f"  {BOLD}{fam.stem}{NC}")
            print(f"    {', '.join(m.name for m in members)}")
    elif args.singletons_included:
        for fam in visible:
            if any(m.role == "umbrella" for m in fam.members):
                continue
            show(fam)
    else:
        # Default: multi-member families only (skip singletons + umbrella)
        for fam in visible:
            if any(m.role == "umbrella" for m in fam.members):
                continue
            if fam.is_singleton:
                continue
            show(fam)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------


def _read_version(m: Module) -> str:
    if not m.package_yaml.exists():
        return "no package.yaml"
    v = get_version(m.package_yaml.read_text())
    return v if v else "no version"


def cmd_status(args: argparse.Namespace) -> int:
    _, modules = discover()
    issues = 0
    if not args.short:
        print(f"{BOLD}{BLUE}=== Git Status ==={NC}\n")

    versions = {name: _read_version(m) for name, m in modules.items()}
    name_w = max((len(n) for n in modules), default=0)

    for name, m in sorted(modules.items()):
        version = versions[name]
        if not m.is_git:
            issues += 1
            if args.short:
                print(f"no-git {name} {version}")
            else:
                print(f"  {YELLOW}no-git{NC}    {name:<{name_w}}  {DIM}{version}{NC}")
            continue

        problems: List[str] = []
        short_flags: List[str] = []

        if git_has_unstaged(m):
            problems.append(f"{RED}modified{NC}")
            short_flags.append("M")
        if git_has_staged(m):
            problems.append(f"{RED}staged{NC}")
            short_flags.append("S")
        if git_untracked(m):
            problems.append(f"{YELLOW}untracked{NC}")
            short_flags.append("U")

        br = git_branch(m)
        if br is None:
            problems.append(f"{YELLOW}detached-HEAD{NC}")
            short_flags.append("D")
        elif br not in DEFAULT_BRANCHES:
            problems.append(f"{YELLOW}branch:{br}{NC}")
            short_flags.append(f"B:{br}")

        ab = git_ahead_behind(m)
        if ab is None:
            if br is not None:
                problems.append(f"{YELLOW}no-upstream{NC}")
                short_flags.append("nu")
        else:
            ahead, behind = ab
            if ahead:
                problems.append(f"{YELLOW}ahead:{ahead}{NC}")
                short_flags.append(f"a{ahead}")
            if behind:
                problems.append(f"{YELLOW}behind:{behind}{NC}")
                short_flags.append(f"b{behind}")

        if not problems:
            if args.short:
                print(f"ok {name} {version}")
            else:
                print(f"  {GREEN}ok{NC}        {name:<{name_w}}  {DIM}{version}{NC}")
        else:
            issues += 1
            if args.short:
                print(f"{','.join(short_flags)} {name} {version}")
            else:
                joined = ", ".join(problems)
                print(f"  {joined}  {name:<{name_w}}  {DIM}{version}{NC}")

    if not args.short:
        print()
        if issues == 0:
            print(f"  {GREEN}All git repos clean.{NC}")
        else:
            print(f"  {YELLOW}{issues} module(s) with issues.{NC}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: test
# ---------------------------------------------------------------------------


def _run_one_test(m: Module) -> Tuple[Module, str, str]:
    """Run morloc tests for one module. Returns (module, status, detail).

    status in {"PASS", "FAIL", "SKIP"}.
    """
    test_loc = m.path / "test.loc"
    if not test_loc.exists():
        return (m, "SKIP", "no test.loc")

    build = run(
        ["morloc", "make", "-o", "test", "test.loc"],
        cwd=m.path,
    )
    if build.returncode != 0:
        return (m, "FAIL", "build error:\n" + build.stderr.strip())

    exe = m.path / "test"
    if not exe.exists():
        return (m, "FAIL", "build produced no `test` binary")

    runp = run([str(exe), "test"], cwd=m.path)
    if runp.returncode != 0:
        return (m, "FAIL", f"test runner exit {runp.returncode}\n{runp.stderr.strip()}")

    last = (runp.stdout.splitlines() or [""])[-1].strip()
    if last == "true":
        return (m, "PASS", "")
    return (m, "FAIL", f"final line was {last!r} (expected 'true')")


def _run_tests(
    selected: List[Module], jobs: int = 1, verbose: bool = False
) -> int:
    """Run morloc tests across the given modules. Returns # of failures."""
    print(f"{BOLD}{BLUE}=== Module Tests ==={NC}\n")
    pass_n = fail_n = skip_n = 0
    failed: List[Tuple[Module, str]] = []

    if jobs and jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(_run_one_test, m): m for m in selected}
            for fut in as_completed(futures):
                m, status, detail = fut.result()
                _print_test_line(m, status, detail)
                if status == "PASS":
                    pass_n += 1
                elif status == "FAIL":
                    fail_n += 1
                    failed.append((m, detail))
                else:
                    skip_n += 1
    else:
        for m in selected:
            print(f"  .... {m.name}", end="\r", flush=True)
            mod, status, detail = _run_one_test(m)
            _print_test_line(mod, status, detail)
            if status == "PASS":
                pass_n += 1
            elif status == "FAIL":
                fail_n += 1
                failed.append((mod, detail))
            else:
                skip_n += 1

    print()
    print(f"  {GREEN}{pass_n} passed{NC}, {RED}{fail_n} failed{NC}, "
          f"{YELLOW}{skip_n} skipped{NC}")

    if failed:
        print()
        print("  Failed modules:")
        for m, detail in failed:
            print(f"    {RED}-{NC} {m.name}")
            if verbose and detail:
                for line in detail.splitlines():
                    print(f"        {DIM}{line}{NC}")
        first = failed[0][0].name
        print()
        print("  Re-run a failed module manually:")
        print(f"    cd {first} && morloc make -o test test.loc && ./test test")

    return fail_n


def cmd_test(args: argparse.Namespace) -> int:
    _, modules = discover()

    # Selection: explicit names, or default to language-specific impls.
    if args.modules:
        selected = []
        for name in args.modules:
            if name not in modules:
                die(f"unknown module: {name!r}")
            selected.append(modules[name])
    else:
        selected = [m for m in modules.values() if m.role == "lang"]
        selected.sort(key=lambda m: m.name)

    return 1 if _run_tests(selected, args.jobs, args.verbose) else 0


def _print_test_line(m: Module, status: str, detail: str) -> None:
    color = {"PASS": GREEN, "FAIL": RED, "SKIP": YELLOW}[status]
    pad = " " * (4 - len(status))
    extra = f"  {DIM}({detail.splitlines()[0]}){NC}" if status == "SKIP" and detail else ""
    print(f"  {color}{status}{NC}{pad} {m.name}{extra}")


# ---------------------------------------------------------------------------
# Subcommand: diff
# ---------------------------------------------------------------------------


def cmd_diff(args: argparse.Namespace) -> int:
    families, modules = discover()
    selected = select_modules(
        families, modules,
        group=args.group, all_=args.all, lang_only=args.lang_only,
        match=args.match,
    )
    any_diff = False
    for m in selected:
        if not m.is_git:
            continue
        cmd = ["diff", *args.paths]
        r = git(cmd, m.path)
        if r.stdout.strip():
            any_diff = True
            print(f"{BOLD}{CYAN}=== {m.name} ==={NC}")
            print(r.stdout)
    if not any_diff:
        print(f"  {DIM}(no diffs in selected modules){NC}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: add
# ---------------------------------------------------------------------------


def cmd_add(args: argparse.Namespace) -> int:
    families, modules = discover()
    selected = select_modules(
        families, modules,
        group=args.group, all_=args.all, lang_only=args.lang_only,
        match=args.match,
    )
    errors, warnings = preflight(selected, PROFILES["add"])
    if errors:
        report_preflight(errors, warnings)
        return 1
    if warnings:
        report_preflight([], warnings)

    # Default to `-A` (everything: tracked + untracked + deletions). Use
    # `-u` if you want only tracked-file modifications.
    pathspec = list(args.paths) if args.paths else ["-A"]

    for m in selected:
        if args.dry_run:
            r = git(["add", "--dry-run", *pathspec], m.path)
            if r.returncode != 0:
                print(f"  {RED}fail{NC}       {m.name}: {r.stderr.strip()}")
                continue
            lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
            if not lines:
                print(f"  {DIM}no-op{NC}      {m.name}: nothing to add")
            else:
                first = lines[0]
                more = f" (+{len(lines) - 1} more)" if len(lines) > 1 else ""
                print(f"  {GREEN}would add{NC}  {m.name}: "
                      f"{DIM}{first}{NC}{more}")
            continue

        # Real mode: compare the staged set before and after to report
        # what was actually staged by this invocation.
        before = git(["diff", "--cached", "--name-only"], m.path).stdout
        before_set = set(before.splitlines())
        r = git(["add", *pathspec], m.path)
        if r.returncode != 0:
            print(f"  {RED}fail{NC}       {m.name}: {r.stderr.strip()}")
            continue
        after = git(["diff", "--cached", "--name-only"], m.path).stdout
        added = set(after.splitlines()) - before_set
        if not added:
            print(f"  {DIM}no-op{NC}      {m.name}: nothing to add")
        else:
            sample = sorted(added)[0]
            more = f" (+{len(added) - 1} more)" if len(added) > 1 else ""
            print(f"  {GREEN}staged{NC}     {m.name}: "
                  f"{DIM}{sample}{NC}{more}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: commit
# ---------------------------------------------------------------------------


def cmd_commit(args: argparse.Namespace) -> int:
    families, modules = discover()
    selected = select_modules(
        families, modules,
        group=args.group, all_=args.all, lang_only=args.lang_only,
        match=args.match,
    )
    errors, warnings = preflight(selected, PROFILES["commit"])
    if errors:
        report_preflight(errors, warnings)
        return 1
    if warnings:
        report_preflight([], warnings)

    fail = 0
    for m in selected:
        if not args.allow_empty and not git_has_staged(m):
            print(f"  {DIM}skip{NC}    {m.name}: no staged changes")
            continue
        cmd = ["commit", "-m", args.message]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.allow_empty:
            cmd.append("--allow-empty")
        r = git(cmd, m.path)
        if r.returncode != 0:
            fail += 1
            print(f"  {RED}fail{NC}    {m.name}: {r.stderr.strip() or r.stdout.strip()}")
            continue
        verb = "would commit" if args.dry_run else "committed"
        print(f"  {GREEN}{verb}{NC}  {m.name}")
    return 1 if fail else 0


# ---------------------------------------------------------------------------
# Subcommand: push
# ---------------------------------------------------------------------------


def cmd_push(args: argparse.Namespace) -> int:
    families, modules = discover()
    selected = select_modules(
        families, modules,
        group=args.group, all_=args.all, lang_only=args.lang_only,
        match=args.match,
    )
    errors, warnings = preflight(
        selected, PROFILES["push"], set_upstream=args.set_upstream
    )
    if errors:
        report_preflight(errors, warnings)
        return 1
    if warnings:
        report_preflight([], warnings)

    fail = 0
    for m in selected:
        cmd = ["push"]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.set_upstream:
            cmd.extend(["-u", "origin", "HEAD"])
        r = git(cmd, m.path)
        if r.returncode != 0:
            fail += 1
            print(f"  {RED}fail{NC}    {m.name}: {r.stderr.strip()}")
            continue
        verb = "would push" if args.dry_run else "pushed"
        print(f"  {GREEN}{verb}{NC}  {m.name}")
        if r.stderr.strip():
            for ln in r.stderr.strip().splitlines():
                print(f"    {DIM}{ln}{NC}")
    return 1 if fail else 0


# ---------------------------------------------------------------------------
# Subcommand: bump
# ---------------------------------------------------------------------------


def cmd_bump(args: argparse.Namespace) -> int:
    families, modules = discover()
    selected = select_modules(
        families, modules,
        group=args.group, all_=args.all, lang_only=args.lang_only,
        match=args.match,
    )
    errors, warnings = preflight(
        selected, PROFILES["bump"], allow_dirty=args.allow_dirty
    )
    if errors:
        report_preflight(errors, warnings)
        return 1
    if warnings:
        report_preflight([], warnings)

    plan: List[Tuple[Module, str, str, str]] = []
    for m in selected:
        text = m.package_yaml.read_text()
        cur = get_version(text)
        if cur is None:
            die(f"{m.name}: no `version:` field in package.yaml")
        new = bump_version_string(cur, args.level)
        plan.append((m, cur, new, text))

    print(f"{BOLD}{BLUE}bump{NC} ({args.level})")
    for m, cur, new, _ in plan:
        marker = "would update" if args.dry_run else "updating"
        print(f"  {marker}  {m.name}: {DIM}{cur}{NC} -> {GREEN}{new}{NC}")

    if args.dry_run:
        return 0

    for m, _, new, text in plan:
        m.package_yaml.write_text(set_version(text, new))

    if args.commit_msg:
        # Stage package.yaml + commit in each member.
        # Members must be git repos and on a non-detached branch.
        commit_errors, _ = preflight(selected, PROFILES["commit"])
        if commit_errors:
            print(f"{YELLOW}note:{NC} bump completed but skipping --commit "
                  f"because some members fail commit pre-flight:")
            report_preflight(commit_errors, [])
            return 0
        for m, _, _, _ in plan:
            r = git(["add", "package.yaml"], m.path)
            if r.returncode != 0:
                print(f"  {RED}stage fail{NC} {m.name}: {r.stderr.strip()}")
                continue
            r = git(["commit", "-m", args.commit_msg], m.path)
            if r.returncode != 0:
                print(f"  {RED}commit fail{NC} {m.name}: {r.stderr.strip()}")
                continue
            print(f"  {GREEN}committed{NC}  {m.name}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: freeze
# ---------------------------------------------------------------------------


def _freezable_modules(
    families: Dict[str, Family], modules: Dict[str, Module]
) -> List[Module]:
    """Every stdlib module except the umbrella itself."""
    return sorted(
        [m for m in modules.values() if m.role != "umbrella"],
        key=lambda m: m.name,
    )


def _read_recorded_pins(text: str) -> Dict[str, str]:
    """Parse the morloc-deps block of the umbrella's package.yaml."""
    block_match = _DEPS_BLOCK_RE.search(text)
    if not block_match:
        return {}
    block = block_match.group(0)
    pins: Dict[str, str] = {}
    name_re = re.compile(r"^\s*-\s*name:\s*(\S+)\s*$")
    hash_re = re.compile(r"^\s*git-hash:\s*(\S+)\s*$")
    cur_name: Optional[str] = None
    for line in block.splitlines():
        m_n = name_re.match(line)
        if m_n:
            cur_name = m_n.group(1)
            continue
        m_h = hash_re.match(line)
        if m_h and cur_name is not None:
            pins[cur_name] = m_h.group(1)
            cur_name = None
    return pins


def cmd_freeze(args: argparse.Namespace) -> int:
    if args.check and args.dry_run:
        die("--check and --dry-run are mutually exclusive")

    families, modules = discover()
    if UMBRELLA_NAME not in modules:
        die(f"missing umbrella module {UMBRELLA_NAME!r}")
    umbrella = modules[UMBRELLA_NAME]
    targets = _freezable_modules(families, modules)

    errors, warnings = preflight(
        targets, PROFILES["freeze"], allow_dirty=args.allow_dirty
    )
    if errors:
        report_preflight(errors, warnings)
        return 1
    if warnings:
        report_preflight([], warnings)

    # Collect current HEAD per target.
    pins: List[Tuple[str, str]] = []
    for m in targets:
        h = git_head_hash(m)
        if not h:
            die(f"{m.name}: could not read HEAD hash")
        pins.append((m.name, h))
    pins.sort(key=lambda p: p[0])

    text = umbrella.package_yaml.read_text()
    new_text = set_morloc_deps(text, pins)

    if args.check:
        recorded = _read_recorded_pins(text)
        actual = dict(pins)
        missing = sorted(set(actual) - set(recorded))
        extra = sorted(set(recorded) - set(actual))
        mismatched = sorted(
            n for n in actual if n in recorded and actual[n] != recorded[n]
        )
        if not (missing or extra or mismatched):
            print(f"  {GREEN}stdlib/package.yaml is in sync with submodule HEADs{NC}")
            return 0
        if missing:
            print(f"  {YELLOW}missing pins{NC} (HEAD seen, not recorded):")
            for n in missing:
                print(f"    - {n}  {DIM}({actual[n][:12]}){NC}")
        if extra:
            print(f"  {YELLOW}stale pins{NC} (recorded, no module on disk):")
            for n in extra:
                print(f"    - {n}  {DIM}({recorded[n][:12]}){NC}")
        if mismatched:
            print(f"  {YELLOW}mismatched pins{NC}:")
            for n in mismatched:
                print(f"    - {n}: {DIM}{recorded[n][:12]}{NC} -> "
                      f"{GREEN}{actual[n][:12]}{NC}")
        return 1

    if args.dry_run:
        print(f"{BOLD}{BLUE}freeze --dry-run{NC} (would write to "
              f"{umbrella.package_yaml})\n")
        # Show only the new block, not the whole file.
        new_block = _DEPS_BLOCK_RE.search(new_text)
        if new_block:
            print(new_block.group(0).rstrip())
        else:
            print("(empty pin set)")
        return 0

    umbrella.package_yaml.write_text(new_text)
    print(f"  {GREEN}wrote{NC}    {umbrella.package_yaml.relative_to(stdlib_root().parent.parent)}")
    print(f"  {DIM}{len(pins)} pinned module(s){NC}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: upload
# ---------------------------------------------------------------------------


def cmd_upload(args: argparse.Namespace) -> int:
    """Pre-flight + tests + add + commit + push as a single chain."""
    families, modules = discover()
    selected = select_modules(
        families, modules,
        group=args.group, all_=args.all, lang_only=args.lang_only,
        match=args.match,
    )

    # Step 1: pre-flight (combined commit + push checks)
    print(f"{BOLD}{BLUE}[1/5]{NC} status check ({len(selected)} module(s))")
    profile = CheckProfile(
        needs_git=True,
        needs_commit=True,
        needs_branch=True,
        needs_upstream=True,
        warns_default_branch=True,
    )
    errors, warnings = preflight(
        selected, profile, set_upstream=args.set_upstream
    )
    if errors:
        report_preflight(errors, warnings)
        die("upload aborted: status checks failed")
    if warnings:
        report_preflight([], warnings)
    print(f"  {GREEN}ok{NC}")

    # Step 2: run the full stdlib language-impl test suite. Even when
    # uploading a single family, downstream tests in other families may
    # depend on it; running the whole suite catches breakage that
    # family-scoped tests would miss.
    _, all_modules = discover()
    test_targets = [m for m in all_modules.values() if m.role == "lang"]
    test_targets.sort(key=lambda m: m.name)
    print(f"\n{BOLD}{BLUE}[2/5]{NC} testing "
          f"{len(test_targets)} language impl(s)")
    fails = _run_tests(test_targets, jobs=args.jobs, verbose=args.verbose)
    if fails:
        die(f"upload aborted: {fails} test(s) failed")

    if args.dry_run:
        print(f"\n{YELLOW}upload --dry-run:{NC} stopping before "
              "add/commit/push (steps 3-5 skipped)")
        return 0

    # Step 3: stage every change in each selected member (tracked +
    # untracked + deletions). `git add -A` is the catch-all.
    print(f"\n{BOLD}{BLUE}[3/5]{NC} staging changes")
    for m in selected:
        before = set(
            git(["diff", "--cached", "--name-only"], m.path).stdout.splitlines()
        )
        r = git(["add", "-A"], m.path)
        if r.returncode != 0:
            die(f"upload aborted: git add failed in {m.name}: "
                f"{r.stderr.strip()}")
        after = set(
            git(["diff", "--cached", "--name-only"], m.path).stdout.splitlines()
        )
        added_n = len(after - before)
        if added_n:
            print(f"  {GREEN}staged{NC}     {m.name} (+{added_n} file(s))")
        else:
            print(f"  {DIM}no-op{NC}      {m.name}: nothing to add")

    # Step 4: commit. Members with nothing staged are skipped silently
    # (consistent with the standalone `commit` subcommand).
    print(f"\n{BOLD}{BLUE}[4/5]{NC} committing with message: "
          f"{DIM}{args.message!r}{NC}")
    committed: List[Module] = []
    for m in selected:
        if not git_has_staged(m):
            print(f"  {DIM}skip{NC}       {m.name}: no staged changes")
            continue
        r = git(["commit", "-m", args.message], m.path)
        if r.returncode != 0:
            die(f"upload aborted: git commit failed in {m.name}: "
                f"{(r.stderr or r.stdout).strip()}")
        print(f"  {GREEN}committed{NC}  {m.name}")
        committed.append(m)

    # Step 5: push every selected member. We push even the ones that
    # had nothing to commit in this run -- a member may have unpushed
    # commits from an earlier session.
    print(f"\n{BOLD}{BLUE}[5/5]{NC} pushing")
    for m in selected:
        cmd = ["push"]
        if args.set_upstream:
            cmd.extend(["-u", "origin", "HEAD"])
        r = git(cmd, m.path)
        if r.returncode != 0:
            die(f"upload aborted: git push failed in {m.name}: "
                f"{r.stderr.strip()}")
        print(f"  {GREEN}pushed{NC}     {m.name}")
        # git push's banner output ("To origin", "branch -> branch") is
        # on stderr. Echo it dimmed so the user sees what the remote did.
        for ln in r.stderr.strip().splitlines():
            print(f"    {DIM}{ln}{NC}")

    print(f"\n{BOLD}{GREEN}upload complete{NC}: "
          f"{len(committed)} commit(s), {len(selected)} push(es)")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


_SELECTOR_BLURB = """\
Group / selector
  GROUP positional   a family stem (e.g. "root", "table") or a single
                     module name (e.g. "root-cpp"). When a stem is given,
                     every member of that family is selected.
  --all              every module discovered under lib/stdlib/, except
                     the `stdlib` umbrella itself.
  --lang-only        restrict to language-specific implementations
                     (modules ending in -cpp / -py / -r / -julia).
  --match REGEX      select every module whose name matches REGEX.

Exactly one selector form must be given.
"""


def _add_selector_args(p: argparse.ArgumentParser) -> None:
    """Common selector flags: GROUP positional, --all, --lang-only, --match."""
    p.add_argument(
        "group", nargs="?", metavar="GROUP",
        help="family stem (e.g. 'root') or single module name (e.g. 'root-cpp')",
    )
    p.add_argument(
        "--all", action="store_true",
        help="select every module except the `stdlib` umbrella",
    )
    p.add_argument(
        "--lang-only", action="store_true",
        help="restrict selection to -cpp / -py / -r / -julia impls",
    )
    p.add_argument(
        "--match", metavar="REGEX",
        help="select every module whose name matches REGEX",
    )


def build_parser() -> argparse.ArgumentParser:
    epilog_main = """\
Examples:
  libman.py groups                       list multi-member families
  libman.py status                       per-module git status
  libman.py test maybe-py -v             test one module, verbose
  libman.py bump root patch --dry-run    preview a patch bump
  libman.py freeze --check               CI: pins match HEADs?

A "family" is a parent type-only module (e.g. `root`) plus its
language-specific siblings (`root-cpp`, `root-py`, `root-r`). Family-scoped
operations (bump / commit / push / add / diff) act on every member at once.

Run any subcommand with --help for its own detailed help and examples.
"""
    p = argparse.ArgumentParser(
        prog="libman.py",
        description=_D(
            "Manage the morloc standard library: discover module families, "
            "run test suites, perform parallel git operations across a family, "
            "and freeze submodule HEADs into the `stdlib` umbrella's pin list."
        ),
        epilog=epilog_main,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp = p.add_subparsers(
        dest="cmd",
        required=True,
        title="subcommands",
        metavar="SUBCOMMAND",
    )

    # ----- groups ------------------------------------------------------
    pg = sp.add_parser(
        "groups",
        help="list discovered module families",
        description=_D(
            "Scan lib/stdlib/ and print the discovered module families. "
            "A family is a parent module (bare stem like `root`) plus its "
            "language-specific siblings (`root-cpp`, `root-py`, `root-r`). "
            "Each member's role is shown in parentheses: (p) parent, "
            "(l) language impl, (s) singleton. Useful for verifying "
            "discovery before running a destructive op."
        ),
        epilog="""\
Examples:
  libman.py groups                        multi-member families only
  libman.py groups --singletons-included  also show 1-member modules
                                                  like `internal`, `regex`
  libman.py groups --lang-only            list only -cpp/-py/-r impls,
                                                  one line per family
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pg.add_argument(
        "--lang-only", action="store_true",
        help="show only language-specific impls (one line per family)",
    )
    pg.add_argument(
        "--singletons-included", action="store_true",
        help="also show 1-member families (default hides them)",
    )
    pg.set_defaults(func=cmd_groups)

    # ----- status ------------------------------------------------------
    ps = sp.add_parser(
        "status",
        help="report git status across every stdlib module",
        description=_D(
            "Walk every stdlib submodule and report its git state: "
            "modified / staged / untracked files, current branch, and "
            "ahead/behind counts vs. upstream. Equivalent to running "
            "`git status` in each submodule, but condensed to one line "
            "per module."
        ),
        epilog="""\
Examples:
  libman.py status            human-readable status report
  libman.py status --short    one-line-per-module, machine-readable
                                      flag set: M=modified, S=staged,
                                      U=untracked, D=detached, B:foo=branch,
                                      a3=3 ahead, b1=1 behind, nu=no upstream
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ps.add_argument(
        "--short", action="store_true",
        help="terse output, one line per module, no header or summary",
    )
    ps.set_defaults(func=cmd_status)

    # ----- test --------------------------------------------------------
    pt = sp.add_parser(
        "test",
        help="run the morloc test suite for stdlib modules",
        description=_D(
            "For each selected module that has a `test.loc` file, run "
            "`morloc make -o test test.loc` and then `./test test`, "
            "checking that the final stdout line is `true`. Modules "
            "without a `test.loc` are skipped. With no arguments, every "
            "language-specific impl (-cpp / -py / -r) is tested."
        ),
        epilog="""\
Examples:
  libman.py test                  test every -cpp/-py/-r module
  libman.py test maybe-py         test a single module
  libman.py test root-py set-cpp  test several modules
  libman.py test -j 4             run up to 4 modules in parallel
  libman.py test maybe-py -v      show build/test stderr on failure
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pt.add_argument(
        "modules", nargs="*", metavar="MODULE",
        help="module names to test (default: every language-specific impl)",
    )
    pt.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N",
        help="run up to N tests in parallel (default: 1)",
    )
    pt.add_argument(
        "-v", "--verbose", action="store_true",
        help="show captured stderr beneath each failing module",
    )
    pt.set_defaults(func=cmd_test)

    # ----- diff --------------------------------------------------------
    pd = sp.add_parser(
        "diff",
        help="show git diff for every member of a family",
        description=_D(
            "Run `git diff` in every member of the selected family and "
            "print the output, prefixed with a banner identifying the "
            "module. Useful for reviewing parallel edits before staging "
            "and committing them with `add` and `commit`. Read-only; no "
            "pre-flight checks."
        ),
        epilog="""\
Examples:
  libman.py diff root             diff root + root-cpp/py/r
  libman.py diff --all            diff every stdlib module
  libman.py diff root -- main.loc only diff main.loc within the family
""" + _SELECTOR_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selector_args(pd)
    pd.add_argument(
        "paths", nargs="*", metavar="PATH",
        help="optional pathspec passed through to `git diff` per member",
    )
    pd.set_defaults(func=cmd_diff)

    # ----- add ---------------------------------------------------------
    pa = sp.add_parser(
        "add",
        help="git add (stage changes) across a family",
        description=_D(
            "Run `git add` in every member of the selected family. With no "
            "pathspec, stages everything: tracked modifications, untracked "
            "files, and deletions (`git add -A`). With a pathspec, passes "
            "it through unchanged. Members with nothing to stage are "
            "reported as `no-op`. Pre-flight: every selected module must "
            "have a `.git` directory."
        ),
        epilog="""\
Examples:
  libman.py add root                       stage every change
                                                   (tracked + untracked)
                                                   in root + root-cpp/py/r
  libman.py add root --dry-run             show what would be staged
  libman.py add root main.loc README.md    stage specific files
""" + _SELECTOR_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selector_args(pa)
    pa.add_argument(
        "paths", nargs="*", metavar="PATH",
        help="pathspec for `git add` (default: `-A`, stage every change)",
    )
    pa.add_argument(
        "--dry-run", action="store_true",
        help="invoke `git add --dry-run`, listing what would be staged",
    )
    pa.set_defaults(func=cmd_add)

    # ----- commit ------------------------------------------------------
    pc = sp.add_parser(
        "commit",
        help="git commit across a family with a shared message",
        description=_D(
            "Run `git commit -m MSG` in every member of the selected family "
            "that has staged changes. Members with nothing staged are "
            "skipped. Pre-flight checks: every selected module must have a "
            "`.git` and not be on a detached HEAD; non-default branches are "
            "warned about but not blocked."
        ),
        epilog="""\
Examples:
  libman.py commit root -m "Add foo"          commit across family
  libman.py commit root -m "msg" --dry-run    preview each commit
  libman.py commit root -m "msg" --allow-empty
                                                       force commit even if
                                                       a member has nothing
                                                       staged
""" + _SELECTOR_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selector_args(pc)
    pc.add_argument(
        "-m", "--message", required=True, metavar="MSG",
        help="commit message used in every member (required)",
    )
    pc.add_argument(
        "--allow-empty", action="store_true",
        help="commit even when a member has no staged changes",
    )
    pc.add_argument(
        "--dry-run", action="store_true",
        help="invoke `git commit --dry-run` per member; nothing is committed",
    )
    pc.set_defaults(func=cmd_commit)

    # ----- push --------------------------------------------------------
    pp = sp.add_parser(
        "push",
        help="git push across a family",
        description=_D(
            "Run `git push` in every member of the selected family. "
            "Pre-flight: every member must have at least one commit on HEAD, "
            "not be on a detached HEAD, and have a tracked upstream (unless "
            "`--set-upstream` is passed). Non-default branches are warned "
            "about but not blocked."
        ),
        epilog="""\
Examples:
  libman.py push root                  push existing branches
  libman.py push root --dry-run        show refs that would be updated
  libman.py push root --set-upstream   push -u origin HEAD per member
                                               (first-time push of a new
                                               branch)
""" + _SELECTOR_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selector_args(pp)
    pp.add_argument(
        "--set-upstream", action="store_true",
        help="`git push -u origin HEAD` per member (sets tracking on push)",
    )
    pp.add_argument(
        "--dry-run", action="store_true",
        help="invoke `git push --dry-run`; nothing is sent to the remote",
    )
    pp.set_defaults(func=cmd_push)

    # ----- bump --------------------------------------------------------
    pb = sp.add_parser(
        "bump",
        help="bump SemVer version field across a family",
        description=_D(
            "Increment the `version:` field in every member's package.yaml "
            "in lockstep, applying a major / minor / patch bump. Pre-flight: "
            "every selected module must have a clean working tree (override "
            "with `--allow-dirty`). Bump does NOT commit by default; pass "
            "`--commit MSG` to stage and commit the version bump in each "
            "member as a single follow-up step."
        ),
        epilog="""\
Examples:
  libman.py bump root patch                  0.1.0 -> 0.1.1 across
                                                     root + root-cpp/py/r
  libman.py bump root minor --dry-run        preview without writing
  libman.py bump root patch --commit \\
      "Bump root family to patch"                    bump and commit in one
                                                     step (each member gets
                                                     its own commit)
  libman.py bump --all minor --allow-dirty   bump every family,
                                                     ignoring dirty trees
""" + _SELECTOR_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selector_args(pb)
    pb.add_argument(
        "level", choices=["major", "minor", "patch"],
        help="which SemVer component to bump",
    )
    pb.add_argument(
        "--allow-dirty", action="store_true",
        help="proceed even if some members have uncommitted changes",
    )
    pb.add_argument(
        "--commit", dest="commit_msg", metavar="MSG",
        help="after bumping, `git add package.yaml && git commit -m MSG` "
             "in each member",
    )
    pb.add_argument(
        "--dry-run", action="store_true",
        help="print the planned old -> new version per member; no writes",
    )
    pb.set_defaults(func=cmd_bump)

    # ----- freeze ------------------------------------------------------
    pf = sp.add_parser(
        "freeze",
        help="write current submodule HEADs into stdlib/package.yaml",
        description=_D(
            "Read `git rev-parse HEAD` for every stdlib module (excluding "
            "the `stdlib` umbrella itself) and write the resulting "
            "(name, hash) set into stdlib/package.yaml under "
            "`morloc-deps`. The result is a deterministic, sorted "
            "pin list that the morloc compiler's resolver consumes when "
            "installing the bundle. Pre-flight: every target module must "
            "have a `.git`, at least one commit, and a clean working tree "
            "(override with `--allow-dirty`)."
        ),
        epilog="""\
Examples:
  libman.py freeze              update stdlib/package.yaml in place
  libman.py freeze --dry-run    print the new pin block to stdout;
                                        no file is written
  libman.py freeze --check      CI mode: report any mismatch between
                                        stdlib's recorded pins and the
                                        actual submodule HEADs and exit
                                        non-zero if anything diverges; no
                                        file is written

`--check` and `--dry-run` are mutually exclusive: `--check` is a binary
yes/no verdict for CI, `--dry-run` is a human-readable preview.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pf.add_argument(
        "--check", action="store_true",
        help="report any pin/HEAD divergence and exit non-zero; no file write",
    )
    pf.add_argument(
        "--dry-run", action="store_true",
        help="print the new pin block to stdout; no file write",
    )
    pf.add_argument(
        "--allow-dirty", action="store_true",
        help="record HEAD even for members with uncommitted changes",
    )
    pf.set_defaults(func=cmd_freeze)

    # ----- upload ------------------------------------------------------
    pu = sp.add_parser(
        "upload",
        help="status + test + add + commit + push, fail-fast",
        description=_D(
            "Combined release-style chain: pre-flight every member of "
            "the selected family, run the full stdlib language-impl "
            "test suite, then stage / commit / push each member with "
            "the same commit message. Stops at the first failing step; "
            "no commits or pushes happen if status checks or tests "
            "fail. Each member with no staged changes is silently "
            "skipped at the commit step but is still pushed (in case "
            "it has earlier unpushed commits)."
        ),
        epilog="""\
Examples:
  libman.py upload root -m "Add foo to root"
                                            full chain on root family
  libman.py upload root -m "msg" --dry-run
                                            run status + tests only;
                                            stop before any add/commit
                                            /push side effect
  libman.py upload root -m "msg" --set-upstream
                                            push -u origin HEAD per
                                            member (first-time push of
                                            new branches)
  libman.py upload --all -m "Sync stdlib" -j 4
                                            chain across every family,
                                            tests parallel-4

Pre-flight requires every selected member to have a `.git`, at least
one commit, a non-detached branch, and (unless --set-upstream is
given) a tracked upstream. Tests run across the full stdlib language
impls, not just the selected family, to catch downstream breakage.
""" + _SELECTOR_BLURB,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selector_args(pu)
    pu.add_argument(
        "-m", "--message", required=True, metavar="MSG",
        help="commit message used in every member (required)",
    )
    pu.add_argument(
        "--set-upstream", action="store_true",
        help="`git push -u origin HEAD` per member during step 5",
    )
    pu.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N",
        help="run up to N tests in parallel during step 2 (default: 1)",
    )
    pu.add_argument(
        "-v", "--verbose", action="store_true",
        help="show captured stderr beneath each failing module",
    )
    pu.add_argument(
        "--dry-run", action="store_true",
        help="run pre-flight and tests, but skip the add/commit/push steps",
    )
    pu.set_defaults(func=cmd_upload)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
