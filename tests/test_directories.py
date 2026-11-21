"""Tracked directories: one manifest entry, but every file inside is synced,
selected and reported on its own."""

import json

import yaml

from syncdot.__main__ import cli


def manifest(sandbox) -> list[dict]:
    return yaml.safe_load((sandbox.repo / "manifest.yaml").read_text())["files"]


def state(sandbox) -> dict:
    return json.loads((sandbox.root / "state" / "state.json").read_text())["files"]


def kitty(sandbox):
    """A tracked ~/.config/kitty with two files."""
    a = sandbox.file(".config/kitty/kitty.conf", "font_size 12\n")
    b = sandbox.file(".config/kitty/themes/dark.conf", "bg black\n")
    sandbox.run("add", "~/.config/kitty")
    return a, b


def test_add_directory_lists_each_file_and_keeps_one_manifest_entry(sandbox):
    sandbox.file(".config/kitty/kitty.conf", "font_size 12\n")
    sandbox.file(".config/kitty/themes/dark.conf", "bg black\n")

    out = sandbox.run("add", "~/.config/kitty").output

    assert "↑  ~/.config/kitty/kitty.conf" in out
    assert "↑  ~/.config/kitty/themes/dark.conf" in out
    assert "↑ 2 pushed" in out
    assert "Tracking ~/.config/kitty as files/.config/kitty (2 files)" in out
    assert manifest(sandbox) == [{"source": "files/.config/kitty", "dest": "~/.config/kitty"}]
    assert sandbox.remote_log()[0] == "add: files/.config/kitty"


def test_add_directory_dry_run_lists_files_and_changes_nothing(sandbox):
    sandbox.file(".config/kitty/kitty.conf", "font_size 12\n")

    out = sandbox.run("add", "~/.config/kitty", "--dry-run").output

    assert "↑  ~/.config/kitty/kitty.conf" in out and "↑ 1 to push" in out
    assert "Tracking ~/.config/kitty as files/.config/kitty (1 file)" in out
    assert manifest(sandbox) in ([], None)
    assert sandbox.repo_dirty() == ""


def test_status_reports_each_file_in_a_directory(sandbox):
    a, b = kitty(sandbox)
    a.write_text("font_size 14\n")

    out = sandbox.run("status").output

    assert "↑  ~/.config/kitty/kitty.conf" in out
    assert "·  ~/.config/kitty/themes/dark.conf" in out
    assert "↑ 1 to push" in out


def test_status_and_diff_of_one_file_inside_a_directory(sandbox):
    a, b = kitty(sandbox)
    a.write_text("font_size 14\n")
    b.write_text("bg white\n")

    out = sandbox.run("status", "~/.config/kitty/kitty.conf", "--diff").output

    assert "~/.config/kitty/kitty.conf" in out and "dark.conf" not in out
    assert "-font_size 12" in out and "+font_size 14" in out


def test_push_and_pull_a_single_file_inside_a_directory(sandbox):
    a, b = kitty(sandbox)
    a.write_text("font_size 14\n")
    b.write_text("bg white\n")

    sandbox.run("push", "~/.config/kitty/kitty.conf")

    assert sandbox.remote_file("files/.config/kitty/kitty.conf") == "font_size 14\n"
    assert sandbox.remote_file("files/.config/kitty/themes/dark.conf") == "bg black\n"
    assert sandbox.remote_log()[0] == "push: files/.config/kitty/kitty.conf"

    sandbox.run("pull", "~/.config/kitty/themes/dark.conf", "--yes")

    assert b.read_text() == "bg black\n"
    assert a.read_text() == "font_size 14\n"


def test_subdirectory_and_glob_select_files_inside_a_directory(sandbox):
    a, b = kitty(sandbox)
    a.write_text("font_size 14\n")
    b.write_text("bg white\n")

    out = sandbox.run("status", "~/.config/kitty/themes").output
    assert "dark.conf" in out and "kitty.conf" not in out

    sandbox.run("push", "~/.config/kitty/*.conf")         # * crosses directories
    assert sandbox.remote_file("files/.config/kitty/themes/dark.conf") == "bg white\n"


def test_changes_on_both_sides_to_different_files_are_not_a_conflict(sandbox):
    """Up to 1.0 a directory was one unit: an edit here and another file's
    edit elsewhere conflicted, and local-wins pushed over the other edit."""
    a, b = kitty(sandbox)
    other = sandbox.other_machine()
    other.write("files/.config/kitty/themes/dark.conf", "bg navy\n")
    other.commit_and_push()
    a.write_text("font_size 14\n")

    out = sandbox.run("sync").output

    assert "conflict" not in out
    assert b.read_text() == "bg navy\n"
    assert sandbox.remote_file("files/.config/kitty/kitty.conf") == "font_size 14\n"
    assert sandbox.remote_file("files/.config/kitty/themes/dark.conf") == "bg navy\n"


def test_conflict_inside_a_directory_only_involves_that_file(sandbox):
    a, b = kitty(sandbox)
    other = sandbox.other_machine()
    other.write("files/.config/kitty/kitty.conf", "font_size 10\n")
    other.write("files/.config/kitty/themes/dark.conf", "bg navy\n")
    other.commit_and_push()
    a.write_text("font_size 14\n")

    out = sandbox.run("sync").output

    assert "!  ~/.config/kitty/kitty.conf" in out and "local-wins" in out
    assert "↓  ~/.config/kitty/themes/dark.conf" in out
    assert sandbox.remote_file("files/.config/kitty/kitty.conf") == "font_size 14\n"


def test_files_added_or_deleted_on_either_side_follow(sandbox):
    a, b = kitty(sandbox)
    new_here = sandbox.file(".config/kitty/extra.conf", "x\n")
    b.unlink()
    other = sandbox.other_machine()
    other.write("files/.config/kitty/remote.conf", "r\n")
    other.commit_and_push()

    out = sandbox.run("sync").output

    assert "↑  ~/.config/kitty/extra.conf" in out
    assert "↑  ~/.config/kitty/themes/dark.conf  [deleted]" in out
    assert "↓  ~/.config/kitty/remote.conf" in out
    assert sandbox.remote_file("files/.config/kitty/extra.conf") == "x\n"
    assert not (sandbox.repo / "files/.config/kitty/themes").exists()   # emptied dir pruned
    assert (sandbox.home / ".config/kitty/remote.conf").read_text() == "r\n"

    other.path.joinpath("files/.config/kitty/remote.conf").unlink()
    other.commit_and_push("delete elsewhere")
    out = sandbox.run("sync").output
    assert "↓  ~/.config/kitty/remote.conf  [deleted]" in out
    assert not (sandbox.home / ".config/kitty/remote.conf").exists()
    assert str(sandbox.home / ".config/kitty/remote.conf") not in state(sandbox)


def test_remove_a_file_inside_a_directory_excludes_it(sandbox):
    (sandbox.repo / "manifest.yaml").write_text("# my dotfiles\nfiles: []\n")
    a, b = kitty(sandbox)

    out = sandbox.run("remove", "~/.config/kitty/themes/dark.conf").output

    assert "−  ~/.config/kitty/themes/dark.conf" in out
    assert ("No longer tracking files/.config/kitty/themes/dark.conf "
            "(excluded from ~/.config/kitty; disk copy kept)") in out
    assert manifest(sandbox) == [{"source": "files/.config/kitty", "dest": "~/.config/kitty",
                                  "exclude": ["themes/dark.conf"]}]
    assert (sandbox.repo / "manifest.yaml").read_text().startswith("# my dotfiles\n")
    assert not (sandbox.repo / "files/.config/kitty/themes").exists()
    assert b.read_text() == "bg black\n"

    # From then on it's ignored, and can't be selected
    b.write_text("bg white\n")
    out = sandbox.run("sync").output
    assert "dark.conf" not in out and "already in sync" in out
    assert "is excluded from tracked directory" in sandbox.run(
        "push", "~/.config/kitty/themes/dark.conf", ok=False).output


def test_add_brings_an_excluded_file_back(sandbox):
    a, b = kitty(sandbox)
    sandbox.run("remove", "~/.config/kitty/themes")
    assert manifest(sandbox)[0]["exclude"] == ["themes"]

    out = sandbox.run("add", "~/.config/kitty/themes").output

    assert "↑  ~/.config/kitty/themes/dark.conf" in out
    assert "Tracking ~/.config/kitty/themes as files/.config/kitty/themes (1 file)" in out
    assert manifest(sandbox) == [{"source": "files/.config/kitty", "dest": "~/.config/kitty"}]
    assert sandbox.remote_file("files/.config/kitty/themes/dark.conf") == "bg black\n"


def test_add_refuses_paths_already_tracked_as_part_of_a_directory(sandbox):
    a, b = kitty(sandbox)
    sandbox.file(".config/other.conf", "o\n")

    assert "already tracked as part of" in sandbox.run("add", str(a), ok=False).output
    assert "contains" in sandbox.run("add", "~/.config", ok=False).output


def test_remove_whole_directory_lists_its_files(sandbox):
    a, b = kitty(sandbox)

    out = sandbox.run("remove", "~/.config/kitty").output

    assert "−  ~/.config/kitty/kitty.conf" in out and "−  ~/.config/kitty/themes/dark.conf" in out
    assert "No longer tracking files/.config/kitty (disk copy kept)" in out
    assert manifest(sandbox) in ([], None)
    assert not any(k.startswith(str(sandbox.home / ".config/kitty")) for k in state(sandbox))


def test_new_machine_pulls_a_single_file_without_setting_up_the_rest(sandbox):
    a, b = kitty(sandbox)
    for f in (a, b):
        f.unlink()
    (sandbox.root / "state" / "state.json").unlink()            # a machine that never synced it

    out = sandbox.run("sync").output
    assert "~/.config/kitty/kitty.conf  [skipped: not synced on this machine yet]" in out
    assert not a.exists()

    sandbox.run("pull", "~/.config/kitty/kitty.conf")
    assert a.read_text() == "font_size 12\n" and not b.exists()
    assert "themes/dark.conf  [skipped: not synced" in sandbox.run("sync").output

    sandbox.run("pull", "~/.config/kitty")                     # the whole directory sets it up
    assert b.exists()
    sandbox.file(".config/kitty/new.conf", "n\n")
    assert "↑  ~/.config/kitty/new.conf" in sandbox.run("sync").output


def test_whole_directory_state_from_1_0_is_migrated(sandbox):
    """Up to 1.0 the state held one hash for the whole directory."""
    from syncdot.state import sha256_dir
    a, b = kitty(sandbox)
    path = sandbox.root / "state" / "state.json"
    data = json.loads(path.read_text())
    kitty_dir = sandbox.home / ".config/kitty"
    data["files"] = {str(kitty_dir): {"last_applied_hash": sha256_dir(kitty_dir),
                                      "last_applied_at": "2026-10-01T00:00:00+00:00"}}
    path.write_text(json.dumps(data))
    other = sandbox.other_machine()
    other.write("files/.config/kitty/themes/dark.conf", "bg navy\n")
    other.commit_and_push()
    a.write_text("font_size 14\n")

    out = sandbox.run("sync").output

    assert "conflict" not in out
    assert "↑  ~/.config/kitty/kitty.conf" in out and "↓  ~/.config/kitty/themes/dark.conf" in out
    files = state(sandbox)
    assert files[str(kitty_dir)]["last_applied_hash"] == "directory"
    assert str(a) in files and str(b) in files


def test_pull_of_a_directory_asks_before_deleting_unsynced_local_files(sandbox):
    a, b = kitty(sandbox)
    local_only = sandbox.file(".config/kitty/scratch.conf", "mine\n")

    result = sandbox.runner.invoke(cli, ["pull", "~/.config/kitty"], input="n\n")
    assert result.exit_code == 1
    assert "~/.config/kitty/scratch.conf" in result.output and local_only.exists()

    out = sandbox.run("pull", "~/.config/kitty", "--yes").output
    assert "↓  ~/.config/kitty/scratch.conf  [deleted]" in out
    assert not local_only.exists()


def test_deleted_here_but_changed_in_the_repo_is_a_conflict(sandbox):
    sandbox.write_config(conflict_resolution="newer-wins")
    a, b = kitty(sandbox)
    other = sandbox.other_machine()
    other.write("files/.config/kitty/kitty.conf", "font_size 10\n")
    other.commit_and_push()
    a.unlink()

    out = sandbox.run("sync").output

    assert "!  ~/.config/kitty/kitty.conf" in out and "newer-wins → repo" in out
    assert a.read_text() == "font_size 10\n"                 # the content is kept
