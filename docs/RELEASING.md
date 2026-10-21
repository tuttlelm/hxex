# Releasing hxex to PyPI

Releases are published by `.github/workflows/publish.yml` when a GitHub release is published.
It uses PyPI "trusted publishing", so no token is stored in the repository.

## One-time setup (reserving the name `hxex`)

A PyPI name is only held once a first release has been uploaded. A "pending publisher" lets
the first publish from GitHub create the project.

1. Create an account on https://pypi.org and enable two-factor authentication.
2. On PyPI, open **Account settings, Publishing** and add a **pending publisher** with:
   - PyPI project name: `hxex`
   - Owner: `tuttlelm`
   - Repository name: `hxex`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. On GitHub, open the repository's **Settings, Environments** and create an environment
   named `pypi`. Optionally require your own approval before it can run.

## Making a release

1. Set `version` in `pyproject.toml` and commit it to `main`.
2. On GitHub, **Releases, Draft a new release**: create a tag `v<version>` (for example
   `v0.0.1.dev0`) targeting `main`, then publish the release.
3. The workflow checks that the tag matches the version, builds the sdist and wheel, runs
   `twine check`, and publishes. The first successful publish creates the PyPI project and
   claims the name.

## Notes

- A version such as `0.0.1.dev0` is a pre-release: `pip install hxex` will not select it
  unless `--pre` is given, so an early placeholder release is not installed by accident.
- Build and check locally before releasing: `uv build` (or `python -m build`), then
  `twine check dist/*`.
- PyPI releases cannot be replaced: a version number can be used only once, and deleting a
  release does not free it.

## The tutorial data

The tutorial (`docs/guide/tutorial.md`) downloads `hxex_b5_tutorial.zip` from a GitHub release
of its own, tagged `tutorial-data-N`. Such a release carries only that file: `publish.yml`
publishes to PyPI only for tags starting with `v`.

1. Build the zip and check it end to end (about 20 minutes on a CPU; needs the `notebook`
   extra): `python docs/tutorial/make_tutorial.py SPECEXPORT_DIR OUT_DIR --run`, where
   `SPECEXPORT_DIR` holds the published B5 SpecExport data. This also redraws the page's
   figures in `docs/images/tutorial_*.png`; commit them if they changed.
2. On GitHub, **Releases, Draft a new release**: create the tag `tutorial-data-N` (the next
   number) on `main`, attach `OUT_DIR/hxex_b5_tutorial.zip`, mark it as a pre-release so it is
   not shown as the latest *hxex* release, and publish.
3. If N changed, update the download link in `docs/guide/tutorial.md`.

A new zip is needed only when the notebook, its README, or its licence change: the data do not.
