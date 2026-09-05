# Security and deployment profiles

## Deployment profiles

The federated learning application has two explicit deployment profiles:

- `development` — the current local Docker/Compose workflow. This profile may use Flower's `--insecure` transport for local integration testing.
- `production` — a fail-closed profile intended for deployment across separate client machines. Production configuration requires an explicit SuperLink address and TLS certificate/key paths.

The default profile is `development` so existing local workflows remain unchanged until the secure transport work in Step 6.2 is implemented.

## Production configuration

Start from `.env.production.example` and provide deployment-specific values on the target host. Never commit the populated `.env` file, TLS private keys, certificates containing private material, or runtime state databases.

The current Step 6.1 validator requires these production variables:

- `DEPLOYMENT_PROFILE=production`
- `SUPERLINK_ADDRESS`
- `TLS_ROOT_CERTIFICATES`
- `SUPERLINK_CERTIFICATE`
- `SUPERLINK_PRIVATE_KEY`

At this stage, the validator establishes the production configuration boundary; it does not yet enable the TLS command-line configuration. That is deliberately deferred to Step 6.2. Production federation must therefore not be started until the TLS transport implementation is complete.

## Certificate/key layout

The intended container-side contract is:

```text
/etc/flower/tls/
├── ca.crt
├── superlink.crt
└── superlink.key
```

The final deployment should use certificates issued by the organization's chosen PKI/certificate authority. Self-signed certificates are suitable only for controlled test environments.

## Secrets handling

Private keys and other runtime credentials belong outside Git. The repository ignores `.env`, certificate/key files, and Flower runtime state under `machine_learning/.flwr/`.

Each production service should receive only the credentials it needs. Per-SuperNode authentication keys will be introduced in Step 6.3.
