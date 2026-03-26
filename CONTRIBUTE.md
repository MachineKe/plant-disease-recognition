# Contributing to plant-disease-recognition

Welcome. This project is actively evolving, and we expect more engineers to join.
These rules keep delivery quality high while minimizing merge risk.

## Core Rules

- Do not push directly to `main`.
- Do not merge a PR without approval.
- Keep PRs focused and small.
- Every user-visible change must include validation evidence.

## Branching Workflow

1. Sync your local main:

```bash
git checkout main
git fetch origin
git pull origin main
npm install
```

2. Create a branch:

```bash
git checkout -b <rf|ft|bg>-#<ticket-id>-<short-title>
```

Example: `ft-#675-add-user-invite-flow`

3. Build and test before pushing:

```bash
npm run build && npm run test
```

4. Commit and push:

```bash
git add -A
git commit -m "<rf|ft|bg>-#<ticket-id>: <short description>"
git push origin <branch>
```

## Pull Request Standards

- PR title example: `[#675] Add onboarding finalize step`
- PR body must include work-item link (Trello/Jira/GitHub Issue)
- Request reviewer(s)
- Assign yourself
- Ensure CI passes before merge

## Required Local Validation

Preferred (Bun):

```bash
bun run typecheck
bun run lint
bun run build
bun run test
```

Equivalent minimum gate:

```bash
npm run build && npm run test
```

## GitHub Settings (Repo Admin)

Configure branch protection for `main`:

- Require pull request before merging
- Require at least 1 approval
- Require status checks to pass before merging
  - `CI / validate`
- Restrict force pushes and branch deletion

## Releases

Releases are SemVer tag-based and automated by GitHub Actions.

See:
- [`RELEASE.md`](./RELEASE.md)
- [`CHANGELOG.md`](./CHANGELOG.md)

Tag examples:
- `v0.4.2` (patch)
- `v0.5.0` (minor)
- `v1.0.0` (major)
