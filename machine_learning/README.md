---
tags: [federated, machine learning, vision, fds]
framework: [torch, torchvision]
---

# Federated Learning with PyTorch and Flower

## Executive Summary

This project implements a **secure, Docker-based Federated Machine Learning framework** for training a shared global model across multiple client devices without centralizing their raw training datasets.

The framework is built around:

- **Flower 1.33.0** for federated orchestration;
- **Python + PyTorch** for machine learning;
- **Docker + Docker Compose** for deployment;

The target operating model is a federation of **two or more client devices**, each maintaining its own local dataset and checkpoint storage. A dedicated server host provides the Flower federation infrastructure, while each client host runs its own authenticated SuperNode and local ClientApp.

The current architecture includes a production-oriented security layer for the Flower control and federation paths:

- TLS protection for the Fleet path;
- TLS protection for the Control path;
- per-client SuperNode authentication;
- server-side public-key registration and authorization state;
- isolation of private client authentication keys.

Runtime/AppIO TLS, secure aggregation, differential privacy, stronger poisoning defenses, expanded audit logging, and additional production hardening remain **planned**; they are not implied by the current TLS and authentication controls.

## Quick navigation

| Goal | Start here |
|---|---|
| Understand the system and current status | `README.md` |
| Deploy locally using multiple terminals or deploy to production | [`DEPLOYMENT.md`](DEPLOYMENT.md) |
| Understand security requirements and trust boundaries | [`SECURITY.md`](SECURITY.md) |
| Understand documentation ownership and source-of-truth rules | [`DOCUMENTATION_GOVERNANCE.md`](DOCUMENTATION_GOVERNANCE.md) |



## Architecture

The system uses Flower's **SuperLink / SuperNode / SuperExec** architecture.

```text
                 +-----------------------+
                 |      SERVER HOST      |
                 +-----------+-----------+
                             |
                    TLS + authentication
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
        CLIENT HOST A  CLIENT HOST B  CLIENT HOST C
              |              |              |
         local data      local data      local data
```

Raw client datasets remain on their respective client hosts. The federation server exchanges the application-defined training results needed to produce the global model rather than requiring the raw datasets to be centralized.

## Data and model contract

All clients use a shared **DataContract** that defines the model-facing data requirements, including:

- required feature columns and label definition;
- feature and label types;
- segmentation length and overlap;
- missing-value handling;
- Continuous Wavelet Transform (CWT) configuration;
- model input tensor shape, layout, and dtype.

The application validates client data against this contract before model training. Incompatible data is rejected rather than silently entering the federation.

## Security architecture

The federation is designed around explicit trust boundaries.

- **Fleet communication:** TLS plus authentication.
- **Control communication:** TLS.
- **Client identity:** one unique identity per client.
- **Private-key isolation:** client private authentication keys remain on the corresponding client host.
- **Authorization:** the server maintains the registered public-key inventory and persistent authorization state.
- **Credential lifecycle:** local development uses generated starter credentials; production uses approved federation credentials.

`SECURITY.md` is the authoritative security architecture and policy document.

## Repository structure

```text
machine_learning/
├── src/                         Application and federation logic
├── tests/                       Automated validation and application tests
├── scripts/                     Deployment and registration helpers
├── clients.yml                  Federation client inventory
├── setup.sh                     Interactive setup/deployment helper
├── Dockerfile.superexec         Shared Flower application runtime image
├── Dockerfile.client-registration
│                                Registration helper image
├── .flwr/config.toml            Flower deployment profile
├── .example.env                 Starting environment template
├── pyproject.toml               Python project metadata
├── requirements.txt             Runtime dependencies for the custom image with machine learning libraries
├── DEPLOYMENT.md                Local distributed and production deployment runbook
├── SECURITY.md                  Security architecture and policy
└── DOCUMENTATION_GOVERNANCE.md  Documentation ownership and maintenance rules
```

### Key application components

- `src/server_app.py` — Flower `ServerApp` and server-side aggregation behavior.
- `src/client_app.py` — Flower `ClientApp` executed for participating clients.
- `src/task.py` — model, preprocessing, training, and local dataset handling.
- `src/data_contract.py` — shared DataContract and validation logic.
- `src/deployment_config.py` — secure deployment configuration validation.
- `scripts/generate_compose.py` — generates server or single-client Docker Compose configuration.
- `scripts/client_registration.py` — registers SuperNode public identities with the Flower Control API.
- `clients.yml` — source of truth for configured federation clients and their server-side public keys.

## Deployment model

### Local distributed development

A single physical machine can simulate a real distributed federation by running the server and each client in separate terminals. The client containers use the machine's LAN-reachable SuperLink address and the same TLS/authentication configuration used in production. Starter credentials are generated automatically when missing.

### Production federation

Production separates the server infrastructure from physical client hosts. The same server/client Compose topology is used, but starter credentials must be replaced with valid federation-approved credentials before production startup.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the complete operational runbook.

## Compatibility and revision policy

The documented deployment baseline is **Flower 1.33.0**, Python 3, and Docker Compose v2. Exact Python/ML dependency versions remain authoritative in the project dependency files and container definitions.

All federation hosts must use the **same Git revision** of the application and deployment scripts.

The generated Flower configuration uses the `production-deployment` profile and derives the Control API endpoint from `SUPERLINK_HOST`. The SuperLink certificate SAN must match that host.

## Project direction

The framework is being developed incrementally toward a production-capable federated learning platform.

The roadmap includes:

1. secure distributed federation foundation;
2. persistent state, retry/failure policy, and health checks;
3. secure aggregation and privacy mechanisms;
4. CI/CD, image registry, scanning, and release processes;
5. observability and audit capabilities;
6. production orchestration and operational hardening.

The architectural goal is to provide a reproducible federation in which **data remains distributed while model training and client identities are centrally but transparently coordinated by the federation server**.
