# Machine-learning documentation governance

This document defines how the documentation under `machine_learning/` is organized, which repository artifacts are authoritative, and how documentation should be maintained as the implementation changes.

## Documentation ownership

The documentation follows a clear separation of concern:

| Document | Owns | Does not own |
|---|---|---|
| `README.md` | Executive summary, architecture, status, navigation, design intent | Detailed deployment commands or security policy |
| `LOCAL_DEPLOYMENT.md` | Single-host development, testing, local federation, cleanup | Production security policy |
| `DISTRIBUTED_DEPLOYMENT.md` | Physical multi-host deployment procedure and acceptance workflow | Authoritative security policy or configuration defaults |
| `SECURITY.md` | Security objectives, trust model, controls, credential policy and limitations | Step-by-step deployment procedure |
| `CONFIGURATION.md` | Configuration reference and semantics | Application behavior or security policy |
| `OPERATIONS.md` | Post-deployment lifecycle and operational procedures | Architecture definition |

The last two documents are planned follow-on documentation; they are not yet present on this branch.

> **README explains. Deployment instructs. Security specifies and constrains.**

## Source of truth

Documentation describes the implementation; it must not become a competing source of configuration or behavior.

Use these repository artifacts as the authoritative sources for the corresponding facts:

| Concern | Authoritative source |
|---|---|
| Python/package dependencies | `pyproject.toml`, `requirements.txt`, Dockerfiles |
| Deployment environment variables and example defaults | `.env.production.example` and related environment examples |
| Deployment profile validation and production security invariants | `src/deployment_config.py` |
| Flower deployment profiles | `.flwr/config.toml` |
| Federation client inventory | `clients.yml` |
| Generated deployment topology | `scripts/generate_compose.py` |
| Client registration behavior | `scripts/client_registration.py` |
| Setup workflow and operator menu | `setup.sh` |
| Application behavior | `src/server_app.py`, `src/client_app.py`, `src/task.py`, `src/data_contract.py` |
| Automated behavior/acceptance | `tests/` |

When an implementation value changes, update the authoritative artifact first and then review the documentation that explains its meaning or operational impact.

## Terminology

Use these terms consistently:

- **SuperLink** — Flower server-side federation infrastructure.
- **SuperNode** — Flower node running on a client host and connecting to the SuperLink Fleet API.
- **ServerApp** — server-side Flower application responsible for orchestration/aggregation behavior.
- **ClientApp** — client-side Flower application responsible for local training.
- **Fleet API** — SuperLink endpoint used by SuperNodes for federated communication (`9092` in the current deployment).
- **Control API** — SuperLink endpoint used by control/registration workflows (`9093` in the current deployment).
- **Runtime/AppIO** — internal Flower execution paths; current deployment uses `9091` and `9094` locally.
- **DataContract** — shared model-facing data/schema contract enforced before local training.
- **Deployment profile** — explicit `development` or `production` configuration selected by `DEPLOYMENT_PROFILE`.
- **Deployment role** — `server`, `client`, or `all`, selected through `DEPLOYMENT_ROLE` where applicable.

Do not use “client” to mean both a physical host and a Flower ClientApp when the distinction matters; use **client host**, **SuperNode**, or **ClientApp** explicitly.

## Status language

Documentation must distinguish implemented behavior from demonstrated behavior and planned work:

- **Implemented** — present in the current code/configuration.
- **Demonstrated** — implemented and exercised in a documented test/deployment scenario.
- **Planned** — intended future engineering work and not currently available.
- **Limitation** — an explicit current boundary that users must account for.

Do not describe planned security or privacy capabilities as if they were already provided by the current TLS/authentication layer.

## Maintenance rules

1. Prefer links to the owning document instead of duplicating long explanations.
2. Avoid hard-coding configuration defaults in multiple Markdown files.
3. Commands in runbooks must correspond to the current scripts and generated service names.
4. Version-sensitive statements should identify their source of truth rather than relying on undocumented assumptions.
5. Security claims must be consistent with `SECURITY.md` and the actual deployment configuration.
6. When a code/configuration change affects an operational workflow, review the relevant runbook in the same change.
7. Keep examples safe to copy: never include real private keys, populated secrets, or deployment-specific credentials.

## Documentation change checklist

Before merging a change that affects the machine-learning deployment:

- [ ] Identify the authoritative implementation/configuration artifact.
- [ ] Identify the documentation owner for the affected behavior.
- [ ] Update the authoritative artifact before documenting a new default.
- [ ] Review cross-document references for stale terminology or commands.
- [ ] Distinguish implemented, demonstrated, planned, and limited behavior.
- [ ] Check that security claims remain consistent with `SECURITY.md`.
- [ ] Check that commands and service names still match `setup.sh` and generated Compose behavior.
- [ ] Avoid introducing duplicated configuration tables unless they serve a distinct audience.
