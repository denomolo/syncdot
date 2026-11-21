"""
sync.py — Core bidirectional sync logic.

For each file in the manifest, the decision matrix is:

  disk_changed | repo_changed | action
  -------------|--------------|------------------------------------------
  False        | False        | nothing (already in sync)
  True         | False        | push: copy disk → repo
  False        | True         | pull: copy repo → disk
  True         | True         | CONFLICT → resolve per conflict_resolution

Conflict resolution modes:
  local-wins   — this machine's version wins (default): the repo's version
                 stays in git history, while a local file that's overwritten
                 is gone
  newer-wins   — newer side wins (disk mtime vs last commit time)
  repo-wins    — the repo's version always wins
"""

from __future__ import annotations

import dataclasses
import difflib
import fnmatch
import functools
import glob
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
import stat
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import yaml

from .config import Config
from .renderer import Renderer, RenderError
from .repo import Repo, GitError
from .state import (DIR_SYNCED, ENV_D_NAME, IGNORED_NAMES, State, is_binary, iter_dir_files, sha256,
                    sha256_dir, sha256_file)


class Action(Enum):
    NOTHING   = auto()
    PUSH      = auto()   # disk → repo
    PULL      = auto()   # repo → disk
    CONFLICT  = auto()
    UNTRACK   = auto()   # removed from tracking (disk file left alone)


# Skip reason for tracked files that were never synced on this machine
NOT_SET_UP = "not synced on this machine yet"


@dataclass
class FileResult:
    dest: Path
    source: str
    action: Action
    resolved_by: Optional[str] = None   # how a conflict was resolved
    error: Optional[str] = None
    skipped: Optional[str] = None       # why the file was skipped (e.g. "binary")
    diff: Optional[list[str]] = None    # unified diff lines, for `status --diff`
    entry: Optional[dict] = None        # the tracked directory the file is inside
    deleted: bool = False               # the file is deleted on the side being written


@dataclass
class Item:
    """
    One synced file: a manifest entry for a file, or a file inside a tracked
    directory (`rel` is its path in there). Files inside directories are
    copied verbatim, and may be missing on either side: deleting one on one
    side deletes it on the other.
    """
    source: str            # path in the repo
    dest: Path             # path on disk
    entry: dict            # its manifest entry
    rel: Optional[str] = None


class PullDeclined(Exception):
    """`pull` stopped because overwriting local changes wasn't confirmed."""


class _PreviewRepo(Repo):
    """A throwaway worktree used by --dry-run: it never switches branches."""

    def checkout(self, branch: str, create: bool = False) -> None:
        pass


def _previewable(method):
    """In --dry-run, run `method` against a preview worktree instead of the clone."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.dry_run or self._preview_root is not None:
            return method(self, *args, **kwargs)
        with self._preview():
            return method(self, *args, **kwargs)
    return wrapper


class Syncer:
    def __init__(self, config: Config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.repo = Repo(config.repo_path)
        self.renderer = Renderer(config.repo_path)
        self.state = State.load(config.state_path, config.profile, config.conflict_resolution)
        self.include_binary = config.include_binary
        self._preview_root: Path | None = None
        self.notes: list[str] = []   # extra things to tell the user after the results
        # What add/checkout started tracking: (disk path, repo path, number of
        # files for a directory), and what remove stopped tracking: (repo
        # path, the directory it's now excluded from, if any)
        self.tracked: list[tuple[Path, str, int | None]] = []
        self.untracked: list[tuple[str, Path | None]] = []
        self._migrate_dir_states()

    # ── Public API ─────────────────────────────────────────────────────────────

    @_previewable
    def sync(self) -> list[FileResult]:
        """Full bidirectional sync: fetch → decide → act → commit → push."""
        branch = self.config.branch

        # 1. Pull remote changes into the repo
        self._git_pull(branch)

        # 2. Process each file in the manifest
        selection = self._select(self.renderer.load_manifest(), None)
        results = self._process_items(selection)

        # 3. Commit whatever changed in the repo: pushes, and conflicts the
        #    machine won (those are reported as CONFLICT, not PUSH)
        if not self.dry_run and self.repo.has_uncommitted():
            if self._git_commit_and_push(branch):
                self._note_unexplained_commit(results)

        # 4. Save updated state
        if not self.dry_run:
            self.state.save()

        # 5. Regenerate the session environment from vars.yaml (env:)
        return results + self._env_results()

    def push_all(self, patterns: list[str] | None = None) -> list[FileResult]:
        """Force disk → repo for all files, or only those matching `patterns` (like local-wins)."""
        branch = self.config.branch
        selection = self._select(self.renderer.load_manifest(), patterns)
        results = [self._push_item(item) for item in _selected_items(selection)]
        if not self.dry_run:
            self._mark_whole_dirs(selection)
            message = None
            if patterns:
                pushed = [r.source for r in results if not r.error and not r.skipped]
                message = (f"push: {pushed[0]}" if len(pushed) == 1
                           else f"push: {len(pushed)} files\n\n" + "\n".join(pushed))
            if self._git_commit_and_push(branch, message=message):
                self._note_unexplained_commit(results)
            self.state.save()
        return results

    @_previewable
    def pull_all(self, patterns: list[str] | None = None,
                 confirm: Callable[[list[Path]], bool] | None = None) -> list[FileResult]:
        """
        Copy the repo's version to disk for all files, or only those matching
        `patterns` (like repo-wins). Local files that would lose content not
        in the repo are listed to `confirm` first; if it returns False (or no
        `confirm` is given), nothing is written and PullDeclined is raised.
        """
        branch = self.config.branch
        self._git_pull(branch)
        selection = self._select(self.renderer.load_manifest(), patterns)
        items = _selected_items(selection)
        at_risk = [item.dest for item in items if self._would_lose_local_changes(item)]
        if at_risk and not self.dry_run and not (confirm and confirm(at_risk)):
            raise PullDeclined("Nothing was pulled.")
        results = [self._pull_item(item) for item in items]
        for r in results:
            if r.dest in at_risk and not r.error and not r.skipped:
                r.resolved_by = "overwrites local changes"
        if not self.dry_run:
            self._mark_whole_dirs(selection)
            self.state.save()
        return results + (self._env_results() if not patterns else [])

    def _would_lose_local_changes(self, item: Item) -> bool:
        """True if writing the repo's version over `item` would lose content
        that isn't in the repo: the file differs from it, and isn't simply the
        copy that was last synced here."""
        dest = item.dest
        if item.rel is not None:
            repo_hash = _file_hash(self.config.repo_path / item.source)
        else:
            try:
                abs_source = self.renderer.resolve_source(item.source)
                if self._is_skipped_binary(abs_source, dest):
                    return False
                repo_hash = sha256(self.renderer.render(item.source))
            except RenderError:
                return False   # reported when pulling
        disk_hash = _file_hash(dest)
        if disk_hash is None or disk_hash == repo_hash:
            return False
        state = self.state.get(dest)
        return disk_hash != (state.last_applied_hash if state else None)

    def status(self, patterns: list[str] | None = None, fetch: bool = True,
               with_diff: bool = False) -> list[FileResult]:
        """
        Report what sync would do for each file (or those matching
        `patterns`), changing nothing. With `fetch`, changes waiting on the
        remote are included, previewed like `sync --dry-run`. With
        `with_diff`, each differing result carries a unified diff.
        """
        old_dry = self.dry_run
        self.dry_run = True
        try:
            if fetch:
                with self._preview():
                    return self._status(patterns, with_diff)
            return self._status(patterns, with_diff)
        finally:
            self.dry_run = old_dry

    def _status(self, patterns: list[str] | None, with_diff: bool) -> list[FileResult]:
        selection = self._select(self.renderer.load_manifest(), patterns)
        results = self._process_items(selection)
        if with_diff:
            for item, r in zip(_selected_items(selection), results):
                if r.action != Action.NOTHING or r.skipped:
                    r.diff = self.diff(item)
        return results + (self._env_results() if not patterns else [])

    def diff(self, item: Item) -> list[str]:
        """
        Unified diff from the repo's version (rendered, for templates) to the
        file on this machine. Empty if they match.
        """
        shown = str(item.dest).replace(str(Path.home()), "~", 1)
        labels = f"repo: {item.source}", f"here: {shown}"
        abs_source = self.config.repo_path / item.source
        if item.rel is not None:
            return _diff_files(abs_source if abs_source.is_file() else None,
                               item.dest if item.dest.is_file() else None, *labels)
        try:
            repo_bytes = self.renderer.render(item.source) if abs_source.exists() else None
        except RenderError as e:
            return [f"(can't render {item.source}: {e})\n"]
        disk_bytes = item.dest.read_bytes() if item.dest.is_file() else None
        return _diff_bytes(repo_bytes, disk_bytes, *labels)

    def add(self, locals_: list[Path], remote: str | None = None,
            allow_private: bool = False) -> list[FileResult]:
        """
        Start tracking one or more local files or directories.

        Each path is copied into files/<remote> (remote defaults to the
        path relative to ~, e.g. ~/.config/foo → files/.config/foo), gets a manifest entry and
        state, then everything is committed and pushed in a single commit.
        With several paths, `remote` is always treated as a directory. A
        path excluded from a tracked directory is tracked as part of it again.
        Directories get one result per file.

        Paths that can't be added (missing, already tracked, binary, private
        unless `allow_private`, …) are returned with `error` set; the rest are
        still added.
        """
        branch = self.config.branch
        as_dir = len(locals_) > 1
        manifest = self.renderer.load_manifest()

        results: list[FileResult] = []
        planned: list[tuple[str, Path, dict | None]] = []
        for local in locals_:
            dest = local.expanduser().absolute()
            try:
                source, parent = self._plan_add(dest, remote, as_dir, manifest, planned, allow_private)
            except ValueError as e:
                results.append(FileResult(dest=dest, source="", action=Action.PUSH, error=str(e)))
                continue
            planned.append((source, dest, parent))
            results.append(FileResult(dest=dest, source=source, action=Action.PUSH))

        if not planned:
            return results

        if not self.dry_run:
            self.repo.checkout(branch)
        done: list[FileResult] = []
        added = []
        for r in results:
            if r.error:
                done.append(r)
                continue
            parent = next(p for s, d, p in planned if d == r.dest)
            if parent is not None:
                # Excluded from a tracked directory: include it again
                rel = r.dest.relative_to(parent["dest"]).as_posix()
                if not self.dry_run:
                    self._edit_manifest_excludes(parent["source"], drop=[rel])
                entry = {**parent, "exclude": [x for x in parent.get("exclude", []) if not _under(x, rel)]}
                items = [i for i in self._items(entry) if _under(i.rel, rel)]
            else:
                entry = {"source": r.source, "dest": str(r.dest)}
                items = self._items(entry)
            file_results = [self._push_item(i, allow_template=True) for i in items]
            is_dir = parent is not None or not items or items[0].rel is not None
            if parent is None and (is_dir or not file_results[0].error) and not self.dry_run:
                self._append_manifest_entry(r.source, r.dest)
                if is_dir:
                    self.state.mark_dir(r.dest)
            if is_dir or not file_results[0].error:
                self.tracked.append((r.dest, r.source, len(items) if is_dir else None))
                added.append(r.source)
            done += file_results or [FileResult(dest=r.dest, source=r.source, action=Action.NOTHING)]
        if added and not self.dry_run:
            message = f"add: {added[0]}" if len(added) == 1 else f"add: {len(added)} files\n\n" + "\n".join(added)
            self._git_commit_and_push(branch, message=message)
            self.state.save()
        return done

    def _plan_add(self, dest: Path, remote: str | None, as_dir: bool,
                  manifest: list[dict], planned: list[tuple[str, Path, dict | None]],
                  allow_private: bool = False) -> tuple[str, dict | None]:
        """
        Validate adding `dest` and return its repo source path, plus the
        tracked directory entry it's excluded from (None for a new entry);
        raise ValueError if it can't be added.
        """
        if not dest.exists():
            raise ValueError(f"Local path not found: {dest}")
        if dest.is_symlink():
            raise ValueError(f"Symlinks are not tracked: {dest}")
        if dest.is_file() and not self.include_binary and is_binary(dest):
            raise ValueError(
                f"{dest} is a binary file; only text files are synced "
                "(set include_binary: true in config to allow binaries)"
            )
        if not allow_private:
            private = find_private(dest, self.include_binary)
            if private is not None:
                where = "is private" if private == dest else f"contains private file {private}"
                raise ValueError(
                    f"{dest} {where} (mode {stat.S_IMODE(private.stat().st_mode):o}), which "
                    "usually means it holds secrets; its contents would be pushed to your "
                    "dotfiles repo (use --allow-private to track it anyway)"
                )

        for other_source, other_dest, _ in planned:
            if other_dest == dest:
                raise ValueError(f"{dest} was given more than once")
        for entry in manifest:
            entry_dest = Path(entry["dest"])
            if entry_dest == dest:
                raise ValueError(f"{dest} is already tracked (source: {entry['source']})")
            if dest.is_relative_to(entry_dest) and self._entry_is_dir(entry):
                rel = dest.relative_to(entry_dest).as_posix()
                excludes = entry.get("exclude", [])
                outer = next((x for x in excludes if _under(rel, x) and x != rel), None)
                if outer is not None:
                    raise ValueError(f"{dest} is inside {entry_dest / outer}, which is excluded "
                                     f"from {entry_dest}; add that instead")
                if not any(_under(x, rel) for x in excludes):
                    raise ValueError(f"{dest} is already tracked as part of {entry_dest}")
                if remote is not None:
                    raise ValueError(f"{dest} is part of tracked directory {entry_dest}, "
                                     "so --remote can't place it elsewhere")
                return f"{entry['source']}/{rel}", entry
            if entry_dest.is_relative_to(dest):
                raise ValueError(f"{dest} contains {entry_dest}, which is already tracked; "
                                 "remove that first")

        files_root = self.config.repo_path / "files"
        if remote is None:
            # Mirror the path under ~ so checkout's default (~/<remote>) maps it back
            if not dest.is_relative_to(Path.home()) or dest == Path.home():
                raise ValueError(f"{dest} is outside your home directory; use --remote to place it")
            rel = dest.relative_to(Path.home()).as_posix()
        else:
            rel = remote.strip("/")
            if not rel or as_dir or remote.endswith("/") or (files_root / rel).is_dir():
                rel = f"{rel}/{dest.name}".lstrip("/")
        repo_source = (files_root / rel).resolve()
        if not repo_source.is_relative_to(files_root.resolve()):
            raise ValueError(f"Remote path escapes files/: {remote}")
        source = f"files/{repo_source.relative_to(files_root.resolve())}"

        for entry in manifest:
            if entry["source"] == source:
                raise ValueError(f"{source} is already used by {entry['dest']}")
        for other_source, other_dest, _ in planned:
            if other_source == source:
                raise ValueError(f"{source} would also be used by {other_dest}; use --remote to separate them")
        if repo_source.exists():
            raise ValueError(f"{source} already exists in the repo")
        return source, None

    @_previewable
    def checkout(self, patterns: list[str], local: Path | None = None,
                 force: bool = False) -> list[FileResult]:
        """
        Start tracking files that already exist in the repo (the reverse of add).

        Each pattern is a path or glob relative to files/. The file is
        written to `local` (default: ~/<path>, minus any .j2 suffix), gets a
        manifest entry and state, then the manifest change is committed and
        pushed. With several files, `local` is always treated as a directory.
        An existing local file with different content is only overwritten
        when `force` is set. Directories get one result per file.
        """
        branch = self.config.branch
        self._git_pull(branch)
        sources = self._expand_sources(patterns)
        as_dir = len(sources) > 1
        manifest = self.renderer.load_manifest()

        results: list[FileResult] = []
        planned: list[tuple[str, Path]] = []
        for source in sources:
            try:
                dest = self._plan_checkout(source, local, as_dir, force, manifest, planned)
            except (ValueError, RenderError) as e:
                results.append(FileResult(dest=local or Path(source), source=source,
                                          action=Action.PULL, error=str(e)))
                continue
            planned.append((source, dest))
            results.append(FileResult(dest=dest, source=source, action=Action.PULL))

        if not planned:
            return results

        if not self.dry_run:
            self.repo.checkout(branch)
        done: list[FileResult] = []
        added = []
        for r in results:
            if r.error:
                done.append(r)
                continue
            entry = {"source": r.source, "dest": str(r.dest)}
            items = self._items(entry)
            file_results = [self._pull_item(i) for i in items]
            is_dir = not items or items[0].rel is not None
            if is_dir or not file_results[0].error:
                if not self.dry_run:
                    self._append_manifest_entry(r.source, r.dest)
                    if is_dir:
                        self.state.mark_dir(r.dest)
                self.tracked.append((r.dest, r.source, len(items) if is_dir else None))
                added.append(r.source)
            done += file_results or [FileResult(dest=r.dest, source=r.source, action=Action.NOTHING)]
        if added and not self.dry_run:
            message = (f"checkout: {added[0]}" if len(added) == 1
                       else f"checkout: {len(added)} files\n\n" + "\n".join(added))
            self._git_commit_and_push(branch, message=message)
            self.state.save()
        return done + self._env_results()

    def _expand_sources(self, patterns: list[str]) -> list[str]:
        """Expand repo paths/globs (relative to files/) into manifest sources."""
        files_root = (self.config.repo_path / "files").resolve()
        sources: dict[str, None] = {}
        for pattern in patterns:
            rel = pattern.strip("/").removeprefix("files/")
            if glob.has_magic(rel):
                matches = {Path(m).relative_to(files_root).as_posix()
                           for m in glob.glob(str(files_root / rel), include_hidden=True)}
                if not matches:
                    raise ValueError(f"Nothing in the repo matches {pattern}")
            else:
                matches = {rel}
            for m in sorted(matches):
                if not (files_root / m).resolve().is_relative_to(files_root):
                    raise ValueError(f"Repo path escapes files/: {pattern}")
                sources.setdefault(f"files/{m}", None)
        return list(sources)

    def _plan_checkout(self, source: str, local: Path | None, as_dir: bool, force: bool,
                       manifest: list[dict], planned: list[tuple[str, Path]]) -> Path:
        """Validate checking out `source` and return its disk path, or raise."""
        abs_source = self.renderer.resolve_source(source)
        if abs_source.is_file() and not self.include_binary and is_binary(abs_source):
            raise ValueError(
                f"{source} is a binary file; only text files are synced "
                "(set include_binary: true in config to allow binaries)"
            )

        name = source.removeprefix("files/").removesuffix(".j2")
        if local is None:
            dest = Path.home() / name
        else:
            dest = Path(os.path.abspath(local.expanduser()))
            if as_dir or str(local).endswith("/") or dest.is_dir() and not abs_source.is_dir():
                dest = dest / Path(name).name

        for entry in manifest:
            if entry["source"] == source:
                raise ValueError(f"{source} is already tracked (dest: {entry['dest']})")
            if source.startswith(entry["source"] + "/"):
                raise ValueError(f"{source} is already tracked as part of {entry['source']}")
            if Path(entry["dest"]) == dest:
                raise ValueError(f"{dest} is already tracked (source: {entry['source']})")
            if dest.is_relative_to(entry["dest"]) or Path(entry["dest"]).is_relative_to(dest):
                raise ValueError(f"{dest} overlaps tracked path {entry['dest']}")
        for other_source, other_dest in planned:
            if other_dest == dest:
                raise ValueError(f"{dest} would also be written by {other_source}")

        if dest.is_symlink():
            raise ValueError(f"{dest} is a symlink; symlinks are not tracked")
        if dest.exists() and not force:
            if abs_source.is_dir() != dest.is_dir():
                raise ValueError(f"{dest} already exists and is a different kind of path (use --force)")
            if abs_source.is_dir():
                same = sha256_dir(abs_source, self.include_binary) == sha256_dir(dest, self.include_binary)
            else:
                same = self.renderer.render(source) == dest.read_bytes()
            if not same:
                raise ValueError(f"{dest} already exists with different content (use --force to overwrite)")
        return dest

    # ── Variables ──────────────────────────────────────────────────────────────

    def var_list(self) -> dict:
        """Template variables in vars.yaml on the active profile's branch."""
        self.repo.checkout(self.config.branch)
        return {k: v for k, v in self.renderer.vars.items() if k != "env"}

    @_previewable
    def var_set(self, values: dict[str, object]) -> list[FileResult]:
        """
        Set variables in vars.yaml on the active profile's branch, re-render
        the templates on disk, then commit and push. Comments are preserved.
        Returns the results for templates whose output changed.
        """
        for name in values:
            _check_var_name(name)
        return self._edit_vars(
            lambda text: _vars_set(text, values),
            message="var: set " + ", ".join(values),
        )

    @_previewable
    def var_unset(self, names: list[str], force: bool = False) -> list[FileResult]:
        """
        Remove variables from vars.yaml on the active profile's branch.
        Refuses if a tracked template still uses one, unless `force`.
        """
        branch = self.config.branch
        self._git_pull(branch)
        current = self.renderer.vars
        missing = [n for n in names if n not in current]
        if missing:
            raise ValueError(f"Not set: {', '.join(missing)}")
        if "env" in names:
            raise ValueError("`env` holds environment variables; use `syncdot env unset`")
        if not force:
            used = {}
            for entry in self.renderer.load_manifest():
                for name in self.renderer.template_variables(entry["source"]) & set(names):
                    used.setdefault(name, []).append(entry["source"])
            for env_name, value in _env_section(current).items():
                for name in self.renderer.string_variables(_env_str(value)) & set(names):
                    used.setdefault(name, []).append(f"env {env_name}")
            if used:
                details = "; ".join(f"{n} by {', '.join(srcs)}" for n, srcs in used.items())
                raise ValueError(f"Still used: {details} (use --force to unset anyway)")
        return self._edit_vars(
            lambda text: _vars_unset(text, names),
            message="var: unset " + ", ".join(names),
            pulled=True,
        )

    def _edit_vars(self, edit, message: str, pulled: bool = False) -> list[FileResult]:
        """Apply `edit` to vars.yaml text, re-render templates, commit and push."""
        branch = self.config.branch
        if not pulled:
            self._git_pull(branch)
        self.repo.checkout(branch)
        vars_path = self.config.repo_path / "vars.yaml"
        old_text = vars_path.read_text() if vars_path.exists() else ""
        _load_vars_text(old_text)
        new_text = edit(old_text)
        new_vars = _load_vars_text(new_text)

        # Templates are rendered from the new values whether or not we write them
        self.renderer._vars = new_vars
        env = self._render_env(new_vars)   # refuse edits that would break the env
        templates = [e for e in self.renderer.load_manifest()
                     if e["source"].endswith(".j2") and not self._entry_is_dir(e)]
        results = [r for r in (self._process_entry(e) for e in templates)
                   if r.action != Action.NOTHING or r.error]
        if self.dry_run:
            return results + self._apply_env(env)

        vars_path.write_text(new_text)
        self._git_commit_and_push(branch, message=message)
        self.state.save()
        return results + self._apply_env(env)

    # ── Environment variables ──────────────────────────────────────────────────

    def env_list(self) -> list[tuple[str, str, str | None, str | None]]:
        """(name, raw value, rendered value, error) for each env: entry."""
        self.repo.checkout(self.config.branch)
        variables = self.renderer.vars
        rows = []
        for name, value in _env_section(variables).items():
            raw = _env_str(value)
            try:
                rows.append((name, raw, self.renderer.render_string(raw, variables), None))
            except RenderError as e:
                rows.append((name, raw, None, str(e)))
        return rows

    @_previewable
    def env_set(self, values: dict[str, str]) -> list[FileResult]:
        """Set environment variables in the env: section of vars.yaml."""
        for name, value in values.items():
            _check_env_name(name)
            if "\n" in value:
                raise ValueError(f"{name}: env values can't contain newlines")
        return self._edit_vars(lambda text: _env_edit(text, set_values=values),
                               message="env: set " + ", ".join(values))

    @_previewable
    def env_unset(self, names: list[str]) -> list[FileResult]:
        """Remove environment variables from the env: section of vars.yaml."""
        self._git_pull(self.config.branch)
        missing = [n for n in names if n not in _env_section(self.renderer.vars)]
        if missing:
            raise ValueError(f"Not set: {', '.join(missing)}")
        return self._edit_vars(lambda text: _env_edit(text, remove=names),
                               message="env: unset " + ", ".join(names), pulled=True)

    @property
    def env_path(self) -> Path:
        return Path.home() / ".config" / "environment.d" / ENV_D_NAME

    def _render_env(self, variables: dict) -> dict[str, str]:
        """Rendered env: section, or raise ValueError if it can't be exported."""
        env = {}
        for name, value in _env_section(variables).items():
            _check_env_name(name)
            try:
                rendered = self.renderer.render_string(_env_str(value), variables)
            except RenderError as e:
                raise ValueError(f"env {name}: {e}") from e
            for bad, what in (("\n", "newlines"), ("$", "$ (systemd would expand it)")):
                if bad in rendered:
                    raise ValueError(f"env {name}: values can't contain {what}")
            env[name] = rendered
        return env

    def _env_results(self) -> list[FileResult]:
        """Regenerate the env files from the current vars, reporting what changed."""
        try:
            env = self._render_env(self.renderer.vars)
        except (ValueError, RenderError) as e:
            return [FileResult(dest=self.env_path, source="vars.yaml env:",
                               action=Action.PULL, error=str(e))]
        return self._apply_env(env)

    def _apply_env(self, env: dict[str, str]) -> list[FileResult]:
        """
        Write ~/.config/environment.d/99-env.conf, read by the systemd user
        session and sourced by shells (see `syncdot env hook`), then update
        the running systemd user manager. Returns a result if it changed.
        """
        path = self.env_path
        text = ("# Generated by syncdot from the env: section of vars.yaml; do not edit.\n"
                "# Shells load it with: set -a; . <this file>; set +a\n"
                + "".join(f'{k}="{_env_quote(v)}"\n' for k, v in env.items())) if env else None
        old = path.read_text() if path.exists() else None
        if text == old:
            return []
        result = FileResult(dest=path, source="vars.yaml env:",
                            action=Action.PULL if text is not None else Action.UNTRACK)
        if self.dry_run:
            return [result]
        old_names = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", old or "", flags=re.M))
        try:
            if text is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".syncdot_tmp")
                tmp.write_text(text)
                tmp.replace(path)
        except OSError as e:
            result.error = str(e)
            return [result]
        _update_systemd_env(env, old_names - env.keys())
        return [result]

    # ── Selecting files ────────────────────────────────────────────────────────

    def _entry_is_dir(self, entry: dict) -> bool:
        """True for an entry that tracks a directory (judged by the repo side first)."""
        source, dest = self.config.repo_path / entry["source"], Path(entry["dest"])
        if source.exists():
            return source.is_dir()
        if dest.exists():
            return dest.is_dir()
        return self.state.dir_synced(dest)

    def _items(self, entry: dict) -> list[Item]:
        """
        The files an entry covers: the file itself, or for a directory every
        file in it on either side, or synced here before (so deletions are
        seen), minus its excludes. Symlinks, nested .git directories and
        (unless include_binary) binary files are left out, as on disk.
        """
        dest = Path(entry["dest"])
        if not self._entry_is_dir(entry):
            return [Item(entry["source"], dest, entry)]
        self._migrate_dir_state(entry)
        roots = (self.config.repo_path / entry["source"], dest)
        rels = set(self.state.paths_under(dest))
        for root in roots:
            if root.is_dir():
                rels.update(f.relative_to(root).as_posix() for f in iter_dir_files(root, self.include_binary))
        excludes = entry.get("exclude", [])
        return [Item(f"{entry['source']}/{rel}", dest / rel, entry, rel) for rel in sorted(rels)
                if not any(_under(rel, x) for x in excludes)
                and not any(_left_out(root, rel, self.include_binary) for root in roots)]

    def _migrate_dir_states(self) -> None:
        """
        Convert directory state from 1.0 and earlier before anything is
        pulled, while the clone still holds what was last synced.
        """
        if not any(v.last_applied_hash != DIR_SYNCED and Path(k).is_dir()
                   for k, v in self.state.files.items()):
            return
        try:
            manifest = self.renderer.load_manifest()
        except (RenderError, yaml.YAMLError):
            return
        for entry in manifest:
            if self._entry_is_dir(entry):
                self._migrate_dir_state(entry)

    def _migrate_dir_state(self, entry: dict) -> None:
        """
        Up to 1.0 a directory had one state hash for all its files. If one
        side still matches it, that side is what was last synced: record each
        of its files as synced. Otherwise only files that match on both sides
        count as synced, and the others are conflicts.
        """
        dest = Path(entry["dest"])
        old = self.state.get(dest)
        if old is None or old.last_applied_hash == DIR_SYNCED:
            return
        for root in (dest, self.config.repo_path / entry["source"]):
            if sha256_dir(root, self.include_binary) == old.last_applied_hash:
                for f in iter_dir_files(root, self.include_binary):
                    self.state.record(dest / f.relative_to(root), f.read_bytes())
                break
        self.state.mark_dir(dest)

    def _select(self, manifest: list[dict], patterns: list[str] | None
                ) -> list[tuple[dict, str | None, list[Item]]]:
        """
        What `patterns` (disk paths) pick out of the manifest, all of it if
        none are given: (entry, rel, items) for each whole entry (rel None) or
        path inside a tracked directory (rel is relative to it), with the
        files it covers.

        Globs are matched against tracked paths, files inside directories
        included, not the filesystem, so files missing from disk can still be
        selected. Raises ValueError if any pattern matches nothing tracked.
        """
        cache: dict[int, list[Item]] = {}
        def items(i: int) -> list[Item]:
            if i not in cache:
                cache[i] = self._items(manifest[i])
            return cache[i]

        if not patterns:
            return [(e, None, items(i)) for i, e in enumerate(manifest)]
        picked: dict[tuple[int, str | None], None] = {}
        for pattern in patterns:
            path = os.path.abspath(os.path.expanduser(pattern))
            found: list[tuple[int, str | None]] = []
            for i, e in enumerate(manifest):
                dest = os.path.abspath(e["dest"])
                if glob.has_magic(path):
                    if fnmatch.fnmatchcase(dest, path):
                        found.append((i, None))
                    else:
                        found += [(i, it.rel) for it in items(i)
                                  if it.rel is not None and fnmatch.fnmatchcase(str(it.dest), path)]
                elif dest == path:
                    found.append((i, None))
                elif Path(path).is_relative_to(dest):
                    rel = Path(path).relative_to(dest).as_posix()
                    if any(it.rel is not None and _under(it.rel, rel) for it in items(i)):
                        found.append((i, rel))
                    elif any(_under(rel, x) for x in e.get("exclude", [])):
                        raise ValueError(f"{pattern} is excluded from tracked directory {e['dest']}")
            if not found:
                raise ValueError(f"{pattern} is not tracked")
            picked.update(dict.fromkeys(found))

        selection = []
        for i, rel in sorted(picked, key=lambda k: (k[0], k[1] or "")):
            # Skip paths that a whole entry or a directory above them covers
            if rel is not None and any(j == i and (r is None or r != rel and _under(rel, r))
                                       for j, r in picked):
                continue
            selection.append((manifest[i], rel,
                              [it for it in items(i) if rel is None or _under(it.rel, rel)]))
        return selection

    def _mark_whole_dirs(self, selection: list[tuple[dict, str | None, list[Item]]]) -> None:
        """Directories selected as a whole are now synced here as a whole."""
        for entry, rel, _ in selection:
            if rel is None and self._entry_is_dir(entry):
                self.state.mark_dir(Path(entry["dest"]))

    def remove(self, patterns: list[str]) -> list[FileResult]:
        """
        Stop tracking files or directories (disk paths or quoted globs matched
        against tracked paths, like push and pull).

        Drops their manifest entries, deletes their sources from the repo,
        forgets their state, then commits and pushes once. A path inside a
        tracked directory is added to the directory's exclude list instead.
        Files on disk are left untouched, and may already be gone.
        """
        branch = self.config.branch
        manifest = self.renderer.load_manifest()
        selection = self._select(manifest, patterns)
        results = []
        for entry, rel, items in selection:
            results += [FileResult(dest=i.dest, source=i.source, action=Action.UNTRACK,
                                   entry=entry if i.rel is not None else None) for i in items]
            if not items:
                results.append(FileResult(dest=Path(entry["dest"]), source=entry["source"],
                                          action=Action.UNTRACK))
            self.untracked.append((entry["source"], None) if rel is None
                                  else (f"{entry['source']}/{rel}", Path(entry["dest"])))
        if self.dry_run:
            return results

        self.repo.checkout(branch)
        whole = [e for e, rel, _ in selection if rel is None]
        remaining = [e for e in manifest if not any(e is w for w in whole)]
        excludes: dict[str, list[str]] = {}
        for entry, rel, items in selection:
            if rel is None:
                self._remove_manifest_entry(entry["source"])
                # Keep the source if another (hand-written) entry still points at it
                if not any(e["source"] == entry["source"] for e in remaining):
                    _delete_path(self.config.repo_path / entry["source"])
                self.state.forget_tree(Path(entry["dest"]))
            else:
                excludes.setdefault(entry["source"], []).append(rel)
                path = self.config.repo_path / entry["source"] / rel
                _delete_path(path)
                _prune_empty_dirs(path.parent, self.config.repo_path / entry["source"])
                for item in items:
                    self.state.forget(item.dest)
        for source, rels in excludes.items():
            self._edit_manifest_excludes(source, add=rels)

        sources = [source for source, _ in self.untracked]
        message = (f"remove: {sources[0]}" if len(sources) == 1
                   else f"remove: {len(sources)} files\n\n" + "\n".join(sources))
        self._git_commit_and_push(branch, message=message)
        self.state.save()
        return results

    # ── Manifest editing ───────────────────────────────────────────────────────

    def _find_manifest_item(self, lines: list[str], source: str) -> tuple[int, int, int]:
        """Line range [start, end) and indent of the list item whose source matches."""
        for i, line in enumerate(lines):
            m = re.match(r"^(\s*)-(\s|$)", line)
            if not m:
                continue
            indent = len(m.group(1))
            end = i + 1
            while end < len(lines):
                nxt = lines[end]
                if (not nxt.strip() or nxt.lstrip().startswith("#")
                        or len(nxt) - len(nxt.lstrip()) <= indent):
                    break
                end += 1
            try:
                item = yaml.safe_load(textwrap.dedent("".join(lines[i:end])))
            except yaml.YAMLError:
                continue
            if (isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict)
                    and item[0].get("source") == source):
                return i, end, indent
        raise ValueError(f"Could not find manifest entry for {source}")

    def _remove_manifest_entry(self, source: str) -> None:
        """Remove the list item whose source matches, editing text to keep comments."""
        manifest_path = self.config.repo_path / "manifest.yaml"
        lines = manifest_path.read_text().splitlines(keepends=True)
        i, end, _ = self._find_manifest_item(lines, source)
        # Also drop the blank separator line `add` puts before entries
        start = i - 1 if i > 0 and not lines[i - 1].strip() else i
        del lines[start:end]

        new_text = "".join(lines)
        # An emptied list parses as null, which the manifest loader rejects
        if (yaml.safe_load(new_text) or {}).get("files") is None:
            new_text = re.sub(r"^files:[ \t]*$", "files: []", new_text, count=1, flags=re.MULTILINE)
        remaining = (yaml.safe_load(new_text) or {}).get("files") or []
        if any(isinstance(e, dict) and e.get("source") == source for e in remaining):
            raise ValueError(f"Could not cleanly remove manifest entry for {source}")
        manifest_path.write_text(new_text)

    def _edit_manifest_excludes(self, source: str, add: list[str] = (), drop: list[str] = ()) -> None:
        """
        Add paths to a directory entry's exclude list, or drop the ones at or
        under `drop` paths, rewriting only that entry so comments elsewhere stay.
        """
        manifest_path = self.config.repo_path / "manifest.yaml"
        lines = manifest_path.read_text().splitlines(keepends=True)
        start, end, indent = self._find_manifest_item(lines, source)
        item = yaml.safe_load(textwrap.dedent("".join(lines[start:end])))[0]
        excludes = [x for x in item.get("exclude") or [] if not any(_under(x, d) for d in drop)]
        excludes += [x for x in add if x not in excludes]
        item.pop("exclude", None)
        if excludes:
            item["exclude"] = sorted(excludes)
        dumped = yaml.safe_dump([item], sort_keys=False)
        lines[start:end] = [" " * indent + line + "\n" for line in dumped.splitlines()]

        new_text = "".join(lines)
        entries = (yaml.safe_load(new_text) or {}).get("files") or []
        if [e.get("exclude") for e in entries if isinstance(e, dict)
                and e.get("source") == source] != [item.get("exclude")]:
            raise ValueError(f"Could not cleanly edit the manifest entry for {source}")
        manifest_path.write_text(new_text)

    def _append_manifest_entry(self, source: str, dest: Path) -> None:
        """Append to manifest.yaml as text so existing comments are preserved."""
        manifest_path = self.config.repo_path / "manifest.yaml"
        text = manifest_path.read_text()
        home = str(Path.home())
        dest_str = "~" + str(dest)[len(home):] if str(dest).startswith(home + "/") else str(dest)

        # Scaffolded manifests start as `files: []`, which can't be appended to
        text = re.sub(r"^files:\s*\[\]\s*$", "files:", text, count=1, flags=re.MULTILINE)
        if not text.endswith("\n"):
            text += "\n"
        entry = yaml.safe_dump([{"source": source, "dest": dest_str}], sort_keys=False)
        text += "\n" + "".join(f"  {line}\n" for line in entry.splitlines())
        manifest_path.write_text(text)

    # ── Per-file logic ─────────────────────────────────────────────────────────

    def _process_items(self, selection: list[tuple[dict, str | None, list[Item]]]) -> list[FileResult]:
        """Decide and act on each selected file, as sync does."""
        results = [self._process_item(item) for item in _selected_items(selection)]
        # A directory never synced here whose files all match already is in
        # sync as a whole, so files that appear in it later are synced too
        for entry, rel, items in selection:
            dest = Path(entry["dest"])
            if (rel is None and items and items[0].rel is not None and not self.dry_run
                    and not self.state.dir_synced(dest)
                    and all(r.action == Action.NOTHING and not r.skipped and not r.error
                            for r in results if r.entry is entry)):
                self.state.mark_dir(dest)
        return results

    def _process_item(self, item: Item) -> FileResult:
        if item.rel is None:
            return self._process_entry(item.entry)
        return self._process_dir_file(item)

    def _process_entry(self, entry: dict) -> FileResult:
        item = Item(entry["source"], Path(entry["dest"]), entry)
        source, dest = item.source, item.dest

        try:
            abs_source = self.renderer.resolve_source(source)
            if self._is_skipped_binary(abs_source, dest):
                return FileResult(dest=dest, source=source, action=Action.NOTHING, skipped="binary")
            rendered = self.renderer.render(source)
            repo_hash = sha256(rendered)
        except RenderError as e:
            return FileResult(dest=dest, source=source, action=Action.NOTHING, error=str(e))

        disk_hash    = _file_hash(dest)
        state_entry  = self.state.get(dest)
        last_hash    = state_entry.last_applied_hash if state_entry else None

        disk_changed = disk_hash != last_hash
        repo_changed = repo_hash != last_hash

        # Never synced on this machine: sync doesn't bring it into play by
        # itself. pull, push or checkout does that explicitly.
        if last_hash is None:
            if disk_hash == repo_hash:
                # Identical already, so nothing changes: just start tracking state
                if not self.dry_run:
                    self.state.record_hash(dest, repo_hash)
                return FileResult(dest=dest, source=source, action=Action.NOTHING)
            return FileResult(dest=dest, source=source, action=Action.NOTHING, skipped=NOT_SET_UP)
        elif not disk_changed and not repo_changed:
            action = Action.NOTHING
        elif disk_hash == repo_hash:
            # Both sides changed identically — just record the new state
            if not self.dry_run:
                self.state.record_hash(dest, repo_hash)
            return FileResult(dest=dest, source=source, action=Action.NOTHING)
        elif disk_changed and not repo_changed:
            action = Action.PUSH
        elif repo_changed and not disk_changed:
            action = Action.PULL
        else:
            action = Action.CONFLICT

        return self._act(item, rendered, action)

    def _process_dir_file(self, item: Item) -> FileResult:
        """
        The same decision for a file inside a tracked directory, where either
        side may be missing (hash None): a file added or deleted on one side
        since the last sync is added or deleted on the other.
        """
        repo_hash = _file_hash(self.config.repo_path / item.source)
        disk_hash = _file_hash(item.dest)
        state = self.state.get(item.dest)
        last_hash = state.last_applied_hash if state else None

        if disk_hash == repo_hash:
            if not self.dry_run:
                if disk_hash is None:
                    self.state.forget(item.dest)
                elif disk_hash != last_hash:
                    self.state.record_hash(item.dest, disk_hash)
            return self._result(item, Action.NOTHING)
        if last_hash is None and not self.state.dir_synced(Path(item.entry["dest"])):
            return self._result(item, Action.NOTHING, skipped=NOT_SET_UP)
        if repo_hash == last_hash:
            action = Action.PUSH       # changed, added or deleted here
        elif disk_hash == last_hash:
            action = Action.PULL       # changed, added or deleted in the repo
        else:
            action = Action.CONFLICT
        return self._act(item, None, action)

    def _result(self, item: Item, action: Action, **kwargs) -> FileResult:
        return FileResult(dest=item.dest, source=item.source, action=action,
                          entry=item.entry if item.rel is not None else None, **kwargs)

    def _act(self, item: Item, rendered: bytes | None, action: Action) -> FileResult:
        if action == Action.NOTHING:
            return self._result(item, action)

        if action == Action.PULL:
            return self._pull_item(item, rendered)

        if action == Action.PUSH:
            return self._push_item(item)

        if action == Action.CONFLICT:
            return self._resolve_conflict(item, rendered)

        raise ValueError(f"Unknown action: {action}")

    # ── Atomic file operations ─────────────────────────────────────────────────

    def _pull_item(self, item: Item, rendered: bytes | None = None) -> FileResult:
        if item.rel is None:
            return self._pull_file(item.source, item.dest, rendered)
        return self._pull_dir_file(item)

    def _push_item(self, item: Item, allow_template: bool = False) -> FileResult:
        if item.rel is None:
            return self._push_file(item.source, item.dest, allow_template)
        return self._push_dir_file(item)

    def _pull_file(self, source: str, dest: Path, rendered: bytes | None = None) -> FileResult:
        """Write rendered repo content to disk."""
        try:
            abs_source = self.renderer.resolve_source(source)
            if self._is_skipped_binary(abs_source, dest):
                return FileResult(dest=dest, source=source, action=Action.NOTHING, skipped="binary")
            if rendered is None:
                rendered = self.renderer.render(source)
        except RenderError as e:
            return FileResult(dest=dest, source=source, action=Action.PULL, error=str(e))

        if self.dry_run:
            return FileResult(dest=dest, source=source, action=Action.PULL)

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically via temp file
        tmp = dest.with_suffix(dest.suffix + ".syncdot_tmp")
        try:
            tmp.write_bytes(rendered)
            # Carry over the executable bit git recorded; other bits stay at
            # the umask default (modes like 600 aren't stored in git)
            if abs_source.stat().st_mode & stat.S_IXUSR:
                mode = tmp.stat().st_mode
                tmp.chmod(mode | (mode & 0o444) >> 2)
            tmp.replace(dest)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            return FileResult(dest=dest, source=source, action=Action.PULL, error=str(e))

        self.state.record(dest, rendered)
        return FileResult(dest=dest, source=source, action=Action.PULL)

    def _pull_dir_file(self, item: Item) -> FileResult:
        """Copy a file inside a tracked directory to disk verbatim, or delete it if the repo has none."""
        repo_file = self.config.repo_path / item.source
        result = self._result(item, Action.PULL, deleted=not repo_file.is_file())
        if item.dest.is_dir():
            result.error = f"{item.dest} is a directory here"
        if self.dry_run or result.error:
            return result
        try:
            if result.deleted:
                item.dest.unlink(missing_ok=True)
                _prune_empty_dirs(item.dest.parent, Path(item.entry["dest"]))
                self.state.forget(item.dest)
            else:
                _copy_file(repo_file, item.dest)
                self.state.record(item.dest, item.dest.read_bytes())
        except OSError as e:
            result.error = str(e)
        return result

    def _push_file(self, source: str, dest: Path, allow_template: bool = False) -> FileResult:
        """
        Copy disk file back into the repo source location.

        A .j2 source is never overwritten (that would replace the template
        with its rendered output) unless `allow_template` is set, as add does
        when creating a new source.
        """
        if not dest.exists():
            return FileResult(dest=dest, source=source, action=Action.PUSH,
                              error=f"Disk file not found: {dest}")

        repo_source = self.config.repo_path / source
        if source.endswith(".j2") and repo_source.is_file() and not allow_template:
            try:
                rendered = self.renderer.render(source)
            except RenderError as e:
                return FileResult(dest=dest, source=source, action=Action.PUSH, error=str(e))
            if dest.is_file() and dest.read_bytes() == rendered:
                return FileResult(dest=dest, source=source, action=Action.NOTHING)
            shown = str(dest).replace(str(Path.home()), "~")
            return FileResult(
                dest=dest, source=source, action=Action.PUSH,
                error=f"rendered from template {source}; edit the template in the repo "
                      f"instead (`syncdot pull {shown}` discards the local change)",
            )
        if self._is_skipped_binary(repo_source, dest):
            return FileResult(dest=dest, source=source, action=Action.NOTHING, skipped="binary")
        if self.dry_run:
            return FileResult(dest=dest, source=source, action=Action.PUSH)

        repo_source.parent.mkdir(parents=True, exist_ok=True)
        try:
            if repo_source.is_dir():
                shutil.rmtree(repo_source)
            shutil.copy2(dest, repo_source)
        except Exception as e:
            return FileResult(dest=dest, source=source, action=Action.PUSH, error=str(e))

        # Update state to reflect current disk content
        self.state.record(dest, repo_source.read_bytes())
        return FileResult(dest=dest, source=source, action=Action.PUSH)

    def _push_dir_file(self, item: Item) -> FileResult:
        """Copy a file inside a tracked directory to the repo, or delete the repo's copy if it's gone here."""
        repo_file = self.config.repo_path / item.source
        result = self._result(item, Action.PUSH, deleted=not item.dest.is_file())
        if repo_file.is_dir():
            result.error = f"{item.source} is a directory in the repo"
        if self.dry_run or result.error:
            return result
        try:
            if result.deleted:
                repo_file.unlink(missing_ok=True)
                _prune_empty_dirs(repo_file.parent, self.config.repo_path / item.entry["source"])
                self.state.forget(item.dest)
            else:
                _copy_file(item.dest, repo_file)
                self.state.record(item.dest, repo_file.read_bytes())
        except OSError as e:
            result.error = str(e)
        return result

    # ── Conflict resolution ────────────────────────────────────────────────────

    def _resolve_conflict(self, item: Item, rendered: bytes | None) -> FileResult:
        mode = self.config.conflict_resolution

        if mode == "local-wins":
            return self._local_wins(item, "local-wins")

        if mode == "repo-wins":
            result = self._pull_item(item, rendered)
            result.resolved_by = "repo-wins"
            return result

        # newer-wins: compare disk mtime vs when the repo side was
        # committed. The clone's own mtime is useless: git rewrites the file
        # when it pulls, so it would always look newer than a local edit.
        # A file deleted here counts as oldest, so its content is kept.
        disk_mtime = _mtime(item.dest)
        repo_mtime = self.repo.last_commit_time(item.source)
        if repo_mtime is None:
            repo_mtime = _mtime(self.config.repo_path / item.source)

        if disk_mtime >= repo_mtime and item.dest.exists():
            result = self._local_wins(item, "newer-wins → local")
        else:
            result = self._pull_item(item, rendered)
            result.resolved_by = "newer-wins → repo"

        result.action = Action.CONFLICT
        return result

    def _local_wins(self, item: Item, how: str) -> FileResult:
        """Push the disk version, noting where the repo's version can be recovered."""
        previous = self.repo.last_commit_id(item.source)
        result = self._push_item(item)
        result.resolved_by = how
        if previous and not result.error and not self.dry_run:
            result.resolved_by += f"; repo version kept in history at {previous}"
        result.action = Action.CONFLICT
        return result

    def _is_skipped_binary(self, *paths: Path) -> bool:
        """True if binaries are excluded and any of the given files is binary."""
        return not self.include_binary and any(is_binary(p) for p in paths)

    # ── Git helpers ────────────────────────────────────────────────────────────

    def _git_pull(self, branch: str) -> None:
        """Pull the profile branch, then merge base into it so shared changes reach every profile."""
        if self._preview_root is not None or self.dry_run:
            return   # dry runs never pull; a preview already contains what this would
        try:
            self.repo.checkout(branch)
            if self.repo.remote_ahead(branch):
                self.repo.pull(branch)
        except GitError as e:
            # Non-fatal: offline sync still works
            print(f"  [warn] Could not pull from remote: {e}")
        if branch == "base":
            return
        try:
            if self.repo.merge_base() and self.config.auto_push:
                self.repo.push(branch)
        except GitError as e:
            print(f"  [warn] Could not merge base into {branch}: {e}")

    @contextmanager
    def _preview(self):
        """
        For --dry-run: build what the clone would look like after pulling (the
        profile branch, plus origin, plus base merged in) in a temporary git
        worktree, and point this Syncer at it. The clone's branches, HEAD and
        files are never touched; only `git fetch` updates remote-tracking refs.
        """
        branch = self.config.branch
        try:
            self.repo.fetch()
        except GitError as e:
            print(f"  [warn] Could not fetch from remote (previewing local state): {e}")
        self.repo._run("worktree", "prune", check=False)

        tmp = Path(tempfile.mkdtemp(prefix="syncdot-preview-"))
        worktree = tmp / "repo"
        if self.repo.current_branch() == branch:
            # Include uncommitted edits in the clone (e.g. a hand-edited
            # vars.yaml), which a real sync would commit
            start = self.repo.snapshot(tmp)
        elif self.repo.branch_exists(branch):
            start = branch
        elif _ref_exists(self.repo, f"origin/{branch}"):
            start = f"origin/{branch}"
        else:
            start = "HEAD"
        self.repo._run("worktree", "add", "--detach", str(worktree), start)
        preview = _PreviewRepo(worktree)
        try:
            refs = [f"origin/{branch}"] + (["origin/base", "base"] if branch != "base" else [])
            for ref in refs:
                if not _ref_exists(preview, ref):
                    continue
                merged = preview._run("-c", "user.name=syncdot", "-c", "user.email=syncdot@localhost",
                                      "merge", "--no-edit", ref, check=False)
                if merged.returncode != 0:
                    preview._run("merge", "--abort", check=False)
                    what = "pulling" if ref == f"origin/{branch}" else f"merging {ref} into {branch}"
                    print(f"  [warn] {what} would conflict; sync would stop there and leave it "
                          f"for you to resolve. Previewing without it.")

            saved = (self.config, self.repo, self.renderer)
            self.config = dataclasses.replace(self.config, repo_path=worktree)
            self.repo, self.renderer = preview, Renderer(worktree)
            self._preview_root = worktree
            try:
                yield
            finally:
                self.config, self.repo, self.renderer = saved
                self._preview_root = None
        finally:
            self.repo._run("worktree", "remove", "--force", str(worktree), check=False)
            shutil.rmtree(tmp, ignore_errors=True)
            self.repo._run("worktree", "prune", check=False)

    def _git_commit_and_push(self, branch: str, message: str | None = None) -> bool:
        """Commit everything in the clone and push it. Returns True if a commit was made."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        committed = False
        try:
            self.repo.checkout(branch)
            self.repo.add_all()
            committed = self.repo.commit(message or f"sync: {timestamp}")
            if committed and self.config.auto_push:
                self.repo.push(branch)
        except GitError as e:
            print(f"  [warn] Git commit/push failed: {e}")
        return committed

    def _note_unexplained_commit(self, results: list[FileResult]) -> None:
        """Mention files the last commit changed that no pushed result accounts for,
        e.g. hand edits in the clone, or files a nested .gitignore kept out before."""
        explained = [r.source for r in results
                     if r.source and not r.error and r.action in (Action.PUSH, Action.CONFLICT)]
        extra = [f for f in self.repo.last_commit_files()
                 if not any(f == src or f.startswith(src + "/") for src in explained)]
        if not extra:
            return
        shown = extra[:5] + ([f"… and {len(extra) - 5} more"] if len(extra) > 5 else [])
        noun = "file" if len(extra) == 1 else "files"
        self.notes.append(f"Also committed changes the repo hadn't recorded yet ({len(extra)} {noun}):\n"
                          + "\n".join(f"    {f}" for f in shown))


# ── Directory helpers ──────────────────────────────────────────────────────────

def _diff_files(repo_file: Path | None, disk_file: Path | None, repo_label: str,
                disk_label: str) -> list[str]:
    return _diff_bytes(repo_file.read_bytes() if repo_file else None,
                       disk_file.read_bytes() if disk_file else None, repo_label, disk_label)


def _diff_bytes(repo: bytes | None, disk: bytes | None, repo_label: str, disk_label: str) -> list[str]:
    """Unified diff lines from the repo's content to the disk's (None = missing)."""
    if repo == disk:
        return []
    if b"\0" in (repo or b"")[:8192] or b"\0" in (disk or b"")[:8192]:
        return [f"Binary files {repo_label} and {disk_label} differ\n"]
    def text(data: bytes | None) -> list[str]:
        return [] if data is None else data.decode("utf-8", errors="replace").splitlines(keepends=True)
    lines = list(difflib.unified_diff(text(repo), text(disk),
                                      repo_label if repo is not None else "/dev/null",
                                      disk_label if disk is not None else "/dev/null"))
    # Mark a missing final newline the way git does, so the line breaks stay readable
    return [l if l.endswith("\n") else l + "\n\\ No newline at end of file\n" for l in lines]


def _ref_exists(repo: Repo, ref: str) -> bool:
    return repo._run("rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0


_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _check_var_name(name: str) -> None:
    if not _VAR_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid variable name {name!r}: use letters, digits and _, "
                         "not starting with a digit")
    if name == "env":
        raise ValueError("`env` holds environment variables; use `syncdot env set`")


# Exporting these would break the session or hijack programs
_PROTECTED_ENV = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "PWD", "OLDPWD", "SHLVL", "TERM",
    "DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_SESSION_ID",
    "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK", "SSH_CONNECTION", "MAIL", "IFS",
}


def _check_env_name(name: str) -> None:
    if not _VAR_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid environment variable name {name!r}: use letters, digits "
                         "and _, not starting with a digit")
    if name in _PROTECTED_ENV or name.startswith(("LD_", "BASH_FUNC_")):
        raise ValueError(f"{name} is managed by your session and can't be set by syncdot")


def _env_section(variables: dict) -> dict:
    env = variables.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("env: in vars.yaml must be a mapping of names to values")
    return env


def _env_str(value: object) -> str:
    """Hand-written YAML values (true, 3) as the strings they'd be in a shell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _env_quote(value: str) -> str:
    r"""
    Escape for a double-quoted value that systemd's environment.d parser and
    POSIX shells read identically: both unescape \\ \" and \` there.
    ($ is refused earlier: systemd would expand it.)
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")


def _update_systemd_env(env: dict[str, str], removed: set[str]) -> None:
    """Best effort: make newly started user services see the change without re-login."""
    if not shutil.which("systemctl"):
        return
    for args in ((["set-environment", *(f"{k}={v}" for k, v in env.items())] if env else None),
                 (["unset-environment", *sorted(removed)] if removed else None)):
        if args:
            try:
                subprocess.run(["systemctl", "--user", *args], capture_output=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass


def _deeper(line: str, indent: str) -> bool:
    """True for a non-blank line indented further than `indent`."""
    return bool(line.strip()) and line.startswith(indent) and line[len(indent):][:1] in (" ", "\t")


def _var_block(lines: list[str], name: str, indent: str = "",
               start: int = 0, stop: int | None = None) -> tuple[int, int] | None:
    """Line range [start, end) of key `name` at `indent` in vars.yaml, or None."""
    stop = len(lines) if stop is None else stop
    n = re.escape(name)
    key = re.compile(rf"""^{re.escape(indent)}(?:{n}|"{n}"|'{n}')\s*:""")
    for i in range(start, stop):
        if key.match(lines[i]):
            end = i + 1
            # The value continues on deeper lines; blank and comment lines in
            # between belong to it, trailing ones don't
            while end < stop and (_deeper(lines[end], indent) or not lines[end].strip()
                                  or lines[end].lstrip().startswith("#")):
                end += 1
            while end > i + 1 and not _deeper(lines[end - 1], indent):
                end -= 1
            return i, end
    return None


def _replace_entry(lines: list[str], block: tuple[int, int], entry: str) -> None:
    """Replace a key's lines, keeping a trailing comment on a one-line value."""
    old_line = lines[block[0]].rstrip("\n")
    m = re.search(r"\s+#[^\n]*$", old_line)
    if (block[1] - block[0] == 1 and m and entry.count("\n") == 1
            and _safe_load_or_none(old_line) == _safe_load_or_none(old_line[:m.start()])):
        entry = entry.rstrip("\n") + m.group(0) + "\n"
    lines[block[0]:block[1]] = [entry]


def _safe_load_or_none(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def _load_vars_text(text: str) -> dict:
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"vars.yaml is not valid YAML: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("vars.yaml must be a mapping of names to values")
    return parsed


def _yaml_entry(name: str, value: object) -> str:
    """One `name: value` line (lists/maps inline), or a block if the value needs it."""
    # YAML reads bare keys like `on`, `no` or `y` as booleans; quote those
    key = name if yaml.safe_load(f"{name}: 0") == {name: 0} else f"'{name}'"
    inline = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=float("inf"))
    inline = inline.removesuffix("\n").removesuffix("\n...").rstrip("\n")
    if "\n" not in inline:
        return f"{key}: {inline}\n"
    return yaml.safe_dump({name: value}, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _vars_set(text: str, values: dict[str, object]) -> str:
    """Set top-level keys in vars.yaml text, keeping comments and order."""
    lines = text.splitlines(keepends=True)
    for name, value in values.items():
        entry = _yaml_entry(name, value)
        block = _var_block(lines, name)
        if block:
            _replace_entry(lines, block, entry)
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(entry)
    new_text = "".join(lines)
    parsed = _load_vars_text(new_text)
    if any(parsed.get(n) != v for n, v in values.items()):
        raise ValueError("Could not cleanly edit vars.yaml; edit it by hand in the repo")
    return new_text


def _env_edit(text: str, set_values: dict[str, str] | None = None,
              remove: list[str] | None = None) -> str:
    """Set or remove keys in the env: section of vars.yaml text, keeping comments."""
    set_values, remove = set_values or {}, remove or []
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    block = _var_block(lines, "env")
    if block and lines[block[0]].split(":", 1)[1].split("#", 1)[0].strip():
        # Inline form (`env: {}` or `env: {A: b}`): rewrite as a block
        current = _env_section(_load_vars_text(lines[block[0]]))
        lines[block[0]:block[1]] = ["env:\n"] + ["  " + _yaml_entry(k, _env_str(v))
                                                 for k, v in current.items()]
        block = _var_block(lines, "env")
    if block is None:
        if not set_values:
            return text
        lines.append("env:\n")
        block = (len(lines) - 1, len(lines))

    start, end = block
    indent = next((re.match(r"[ \t]+", l).group(0) for l in lines[start + 1:end]
                   if _deeper(l, "") and not l.lstrip().startswith("#")), "  ")
    for name in remove:
        b = _var_block(lines, name, indent, start + 1, end)
        if b:
            del lines[b[0]:b[1]]
            end -= b[1] - b[0]
    for name, value in set_values.items():
        entry = indent + _yaml_entry(name, value)
        b = _var_block(lines, name, indent, start + 1, end)
        if b:
            _replace_entry(lines, b, entry)
            end -= b[1] - b[0] - 1
        else:
            lines.insert(end, entry)
            end += 1

    new_text = "".join(lines)
    env = _env_section(_load_vars_text(new_text))
    if any(_env_str(env.get(n)) != v or n not in env for n, v in set_values.items()) \
            or any(n in env for n in remove):
        raise ValueError("Could not cleanly edit vars.yaml; edit it by hand in the repo")
    return new_text


def _vars_unset(text: str, names: list[str]) -> str:
    """Remove top-level keys from vars.yaml text, keeping everything else."""
    lines = text.splitlines(keepends=True)
    for name in names:
        block = _var_block(lines, name)
        if block:
            del lines[block[0]:block[1]]
    new_text = "".join(lines)
    if any(n in _load_vars_text(new_text) for n in names):
        raise ValueError("Could not cleanly edit vars.yaml; edit it by hand in the repo")
    return new_text


def find_private(path: Path, include_binary: bool = False) -> Path | None:
    """
    First path that group and others can't access at all (e.g. 600, 700):
    `path` itself, or a synced file inside it if it's a directory.
    """
    if not path.stat().st_mode & 0o077:
        return path
    if path.is_dir():
        for f in iter_dir_files(path, include_binary):
            if not f.stat().st_mode & 0o077:
                return f
    return None


def _mtime(path: Path) -> float:
    """mtime of a file (0 if missing)."""
    return path.stat().st_mtime if path.exists() else 0


def _file_hash(path: Path) -> str | None:
    """sha256 of a regular file's contents, or None if there's no file there."""
    return sha256_file(path) if path.is_file() else None


def _selected_items(selection: list[tuple[dict, str | None, list[Item]]]) -> list[Item]:
    """The files of a selection, each once."""
    items: dict[Path, Item] = {}
    for _, _, its in selection:
        for item in its:
            items.setdefault(item.dest, item)
    return list(items.values())


def _under(rel: str, prefix: str) -> bool:
    """True if relative path `rel` is `prefix` or inside it."""
    return rel == prefix or rel.startswith(prefix + "/")


def _left_out(root: Path, rel: str, include_binary: bool) -> bool:
    """
    True if `rel` inside directory `root` isn't synced: it's in or under an
    ignored name or a symlink (never followed or written through), or a
    binary file while binaries are excluded.
    """
    parts = rel.split("/")
    if IGNORED_NAMES.intersection(parts):
        return True
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            return True
    return not include_binary and path.is_file() and is_binary(path)


def _copy_file(src: Path, dst: Path) -> None:
    """Copy a file with its mode, atomically replacing dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".syncdot_tmp")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dst)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _delete_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _prune_empty_dirs(path: Path, root: Path) -> None:
    """Remove `path` and its parents while they're empty, up to (not including) `root`."""
    while path != root and path.is_relative_to(root) and path.is_dir() \
            and not path.is_symlink() and not any(path.iterdir()):
        path.rmdir()
        path = path.parent
