# Security and deployment profiles

## Deployment profiles

The federated learning application has two explicit deployment profiles:

- `development` — the current local Docker/Compose workflow. This profile may use Flower's `--insecure` transport for local integration testing.
- `production` — the secure deployment profile. Production requires an explicit SuperLink address and TLS certificate/key paths and rejects `--insecure` for the SuperLink/SuperNode federation transport.

The default profile is `development` so existing local workflows remain unchanged.

## TLS infrastructure

Production federation TLS uses a CA certificate to verify the SuperLink and a SuperLink certificate/private key to identify the SuperLink:

```text
/etc/flower/tls/
├── ca.crt
├── superlink.crt
└── superlink.key
```

The SuperLink receives all three files. SuperNodes receive only `ca.crt` and connect with Flower's `--root-certificates` option. The SuperLink private key must never be distributed to SuperNodes.

For controlled development testing, `scripts/generate_dev_certs.py` can create a local CA and SuperLink certificate. Generated material is stored under `certificates/`, which is ignored by Git. Production certificates must come from the organization's chosen PKI/certificate authority.

The development generator includes SANs for `superlink`, `localhost`, and `127.0.0.1`. If a production SuperLink is addressed through another DNS name or IP address, its production certificate must contain the corresponding SAN.

## Docker deployment

Production Compose generation is explicit:

```bash
DEPLOYMENT_PROFILE=production \
SUPERLINK_ADDRESS=fl.example.internal:9092 \
TLS_ROOT_CERTIFICATES=/etc/flower/tls/ca.crt \
SUPERLINK_CERTIFICATE=/etc/flower/tls/superlink.crt \
SUPERLINK_PRIVATE_KEY=/etc/flower/tls/superlink.key \
TLS_CERTIFICATE_HOST_DIR=/etc/federatedhomes/flower/tls \
python generate_compose.py --profile production
```

The production generator mounts the host certificate directory read-only and replaces the SuperLink/SuperNode `--insecure` flags with TLS arguments. The Flower CLI used by the trainer also selects the `production-deployment` profile, which supplies the CA certificate.

Flower containers use a non-root user, so mounted certificate files must be readable by the container user. On Linux, Flower documents UID `49999` for its containers.

## Production configuration

Start from `.env.production.example` and provide deployment-specific values on the target host. Never commit the populated `.env` file, TLS private keys, or runtime state databases.

The production validator requires:

- `DEPLOYMENT_PROFILE=production`
- `SUPERLINK_ADDRESS`
- `TLS_ROOT_CERTIFICATES`
- `SUPERLINK_CERTIFICATE`
- `SUPERLINK_PRIVATE_KEY`

## Secrets handling

Private keys and other runtime credentials belong outside Git. The repository ignores `.env`, certificate/key files, the local `certificates/` tree, and Flower runtime state under `machine_learning/.flwr/`.

Each production service should receive only the credentials it needs. Per-SuperNode authentication keys will be introduced in Step 6.3.

## Scope boundary

Step 6.2 establishes TLS for the SuperLink ↔ SuperNode federation transport and the Flower CLI connection used by the deployment runtime. Internal Runtime API TLS (`--appio-ssl-*`) requires separate per-service certificate/SAN handling and is intentionally kept as a distinct hardening item rather than sharing the SuperLink private key across services. SuperNode authentication is implemented in Step 6.3.
