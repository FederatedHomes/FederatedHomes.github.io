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
python scripts/generate_compose.py --profile production
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

Each production service should receive only the credentials it needs.

### SuperNode authentication

Production SuperNode authentication is enabled with Flower's `--enable-supernode-auth` option on the SuperLink. Each authorized SuperNode has a unique ECDSA P-384 key pair in SSH/OpenSSH format. The public key is registered with the SuperLink; the corresponding private key remains with that SuperNode.

The repository's development helper can generate local authentication identities:

```bash
python scripts/generate_supernode_auth.py \
    --output-dir certificates/dev/auth \
    client-1 client-2 client-3
```

This helper is for development and integration testing only. Production identities must be generated and managed through the organization's approved secret-management/PKI process.

For production, the authentication paths are separated into host and container paths:

```text
Host:
<auth-host-dir>/
├── client-1
├── client-2
└── client-3

Container:
/etc/flower/auth/
├── client-1
├── client-2
└── client-3
```

`SUPERNODE_AUTH_HOST_DIR` identifies the host directory containing the private keys. `SUPERNODE_AUTH_PRIVATE_KEY_DIR` identifies the container directory used by Flower. The Compose generator mounts each SuperNode's key individually and read-only; a SuperNode must never receive the private keys belonging to other clients.

Production configuration requires:

```text
SUPERNODE_AUTH_PRIVATE_KEY_DIR=/etc/flower/auth
SUPERNODE_AUTH_HOST_DIR=<host authentication directory>
```

These variables are required in addition to the TLS configuration. `setup.sh` validates that the configured authentication directory exists and that every client in `clients.yml` has its own authentication private key before production Compose generation.

#### Register authorized SuperNodes

Generating a key pair does not authorize a SuperNode. Register each public key with the configured SuperLink:

```bash
flwr supernode register \
    certificates/prod/auth/client-1.pub \
    production-deployment

flwr supernode register \
    certificates/prod/auth/client-2.pub \
    production-deployment

flwr supernode register \
    certificates/prod/auth/client-3.pub \
    production-deployment
```

Verify the registered identities with:

```bash
flwr supernode list production-deployment
```

Only public keys belonging to authorized SuperNodes should be registered. Record the Node ID returned during registration.

Flower registration expects an OpenSSH ECDSA public key, such as:

```text
ecdsa-sha2-nistp384 AAAA...
```

Do not use a PEM public-key file beginning with `-----BEGIN PUBLIC KEY-----`.

#### Validate authentication keys

Private-key material must never be printed or committed. To validate a private key without displaying it:

```bash
ssh-keygen -y -f certificates/prod/auth/client-1 > /dev/null
```

Repeat for every configured SuperNode.

To compare public-key fingerprints:

```bash
for client in client-1 client-2 client-3; do
    echo "$client:"
    ssh-keygen -lf "certificates/prod/auth/$client.pub"
done
```

Each configured client must have a unique fingerprint.

#### Production authentication flow

The expected production flow is:

```text
SuperNode private key
        |
        v
SuperNode establishes TLS to SuperLink
        |
        v
SuperNode authentication
        |
   +----+----+
   |         |
accepted   rejected
   |         |
   v         v
federated   no access
training
```

TLS protects the transport and verifies the SuperLink using the configured CA. SuperNode authentication separately determines whether the connecting SuperNode identity is authorized.

An authorized SuperNode must connect successfully, while an unregistered SuperNode using a different key must be rejected.

#### Key rotation and revocation

If a SuperNode authentication private key is compromised:

1. Stop the affected SuperNode.
2. Unregister/revoke its old public-key identity from the SuperLink.
3. Generate a new unique key pair through the approved credential-management process.
4. Distribute only the new private key to the affected SuperNode.
5. Register the new public key.
6. Verify the new identity before resuming training.

Do not replace a registered key without considering the impact on the existing Node ID and authorization state.

## Scope boundary

Step 6.2 establishes TLS for the SuperLink ↔ SuperNode federation transport and the Flower CLI connection used by the deployment runtime. Internal Runtime API TLS (`--appio-ssl-*`) requires separate per-service certificate/SAN handling and is intentionally kept as a distinct hardening item rather than sharing the SuperLink private key across services. SuperNode authentication is implemented in Step 6.3.
