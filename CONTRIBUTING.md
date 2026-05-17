# Contributing to cardiometabolic-graph

Thanks for considering a contribution. This project is a public reference
implementation for cardiometabolic patient modeling; we welcome issues, fixes,
and extensions from clinical researchers, ML engineers, and digital
therapeutics teams.

## Quick start for contributors

```bash
git clone https://github.com/amitshenoy/cardiometabolic-graph.git
cd cardiometabolic-graph
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
make up           # boots Postgres + Neo4j + Jupyter
make pipeline     # generates synthetic data, runs ETL, trains, evaluates
```

## Ground rules

1. **No real patient data in commits.** Anything under `data/raw/` is
   gitignored. If you discover a leak, open a security issue immediately.
2. **Reproducibility before novelty.** A PR that improves a model's score but
   silently breaks `make pipeline` will not be merged.
3. **Tests for ETL and schema changes.** Any change to `etl/` or `schema/`
   must come with corresponding tests in `tests/`.
4. **Document the *why*.** README, notebooks, and cookbook entries should
   explain motivation so a stranger landing here from Google can follow.

## Style

- Python: `ruff` + `black` enforced by `pre-commit`. Run `make format` before
  pushing.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `build:`.
- PR titles: same convention. Include a short *why* in the description.

## Branching

- `main` is always green.
- Feature branches: `feat/<short-name>`. Rebase before merge.
- Squash on merge; the squash commit message is the PR title + body.

## Adding a cookbook example

Real-world worked examples live in `cookbook/`. A good cookbook entry:

- Frames a concrete clinical or product question.
- Pulls only from data the repo can actually produce (synthetic or demo).
- Runs end-to-end in under five minutes on a laptop.
- Saves any figures into `cookbook/<example>/figures/`.

See `cookbook/01_at_risk_cohort/` for the canonical template.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For security
concerns, please email the maintainer directly rather than opening a public
issue.

## Code of conduct

All contributors agree to the [Contributor Covenant](CODE_OF_CONDUCT.md).
