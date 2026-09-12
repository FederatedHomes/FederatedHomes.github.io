# Deployment

This is the operational runbook for the Flower 1.33.0 / PyTorch federated learning application.

The framework architecture includes: a server host that runs the Flower SuperLink, ServerApp, trainer, and client-registration service; each client host runs exactly one SuperNode and one ClientApp.

The difference between local development and production is the credential material:

- **Local distributed development:** `setup.sh` generates starter TLS and SuperNode authentication credentials.
- **Production:** starter credentials are replaced by valid federation-approved credentials before deployment.

The topology, Docker Compose files, Flower profile, TLS settings, authentication, registration workflow, and network model remain the same.

## 1. Architecture

```text
                         SERVER HOST
                 +-----------------------+
                 |       SuperLink       |
                 | Fleet API   :9092     |
                 | Control API :9093      |
                 | Runtime     :9091     |
                 +-----------+-----------+
                             |
                 TLS + SuperNode authentication
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
      CLIENT 1           CLIENT 2           CLIENT 3
      SuperNode          SuperNode          SuperNode
      ClientApp          ClientApp          ClientApp
      local data         local data         local data
```

The server Compose file contains:

- `superlink`
- `superexec-serverapp`
- `trainer`
- `client-registration`

Each client Compose file contains exactly:

- one `SuperNode` for the selected client ID;
- one `ClientApp` for that client ID.

Only two host roles are supported: `server` and `client`.

## 2. Configuration

From `machine_learning/`, create the environment file once:

```bash
cp .example.env .env
```

`.env` is the deployment environment template. Do not share/commit a populated `.env`.

The deployment profile is always:

```dotenv
DEPLOYMENT_PROFILE=production
```

The profile name describes the secure Flower deployment configuration, not the provenance of the credentials. Local development uses the same profile with starter credentials.

### Server host

```dotenv
DEPLOYMENT_ROLE=server
SUPERLINK_HOST=<server-lan-dns-or-ip>
```

### Client host

```dotenv
DEPLOYMENT_ROLE=client
CLIENT_ID=client-1
SUPERLINK_HOST=<server-lan-dns-or-ip>
```

The framework owns Flower ports:

| Port | Purpose |
|---:|---|
| 9091 | SuperLink Runtime API; Docker-internal server communication |
| 9092 | SuperLink Fleet API; SuperNode connections |
| 9093 | SuperLink Control API; registration/trainer control |
| 9094 | local SuperNode ClientApp API; Docker-internal |

Do not configure a physical client to use the Docker-only hostname `superlink`.

## 3. Client inventory

`clients.yml` is the federation inventory and is shared by the deployment tooling.

Example:

```yaml
clients:
  - id: client-1
    data_dir: ./data/client-1
    checkpoint_dir: ./checkpoints/client-1
    public_key: ./certificates/prod/auth/client-1.pub

  - id: client-2
    data_dir: ./data/client-2
    checkpoint_dir: ./checkpoints/client-2
    public_key: ./certificates/prod/auth/client-2.pub

  - id: client-3
    data_dir: ./data/client-3
    checkpoint_dir: ./checkpoints/client-3
    public_key: ./certificates/prod/auth/client-3.pub
```

At least two clients are required. Client IDs must be unique.

The public key belongs in the server-side authorization inventory. A client's private authentication key must remain only on the client host assigned to that identity.

## 4. Local distributed development

Local development deliberately uses multiple terminals so that the developer exercises the same distributed topology used in production.

All terminals use the same repository checkout and the same `.env` structure. The server and clients communicate through the host's LAN-reachable address, even when all containers are running on one physical machine.

### Terminal 1 — prepare and start the server

Set:

```dotenv
DEPLOYMENT_ROLE=server
SUPERLINK_HOST=<this-machine-lan-ip-or-dns>
```

Then:

```bash
./setup.sh
```

Select `Prepare host`, then `Start server infrastructure`.

Setup creates starter TLS material when it is missing:

```text
certificates/prod/tls/
├── ca.crt
├── superlink.crt
└── superlink.key
```

It also creates persistent SuperLink state under the configured state directory.

Starter TLS credentials are for controlled development/testing only. They are not production PKI credentials.

### Terminal 2 — client 1

Set:

```dotenv
DEPLOYMENT_ROLE=client
CLIENT_ID=client-1
SUPERLINK_HOST=<server-lan-ip-or-dns>
```

Copy the server's `ca.crt` into the client's configured TLS directory if the client is a separate host. On one physical development machine, the generated starter CA is already available to the client setup.

Run:

```bash
./setup.sh
```

Select `Prepare host`, then `Start client infrastructure`.

Setup creates the starter authentication identity if it is missing:

```text
certificates/prod/auth/
├── client-1
└── client-1.pub
```

The private key is used only by client-1.

### Terminal 3 — client 2

Use another terminal with:

```dotenv
DEPLOYMENT_ROLE=client
CLIENT_ID=client-2
SUPERLINK_HOST=<server-lan-ip-or-dns>
```

Run the same client preparation/start procedure.

Repeat for additional clients.

### Registration

The server-side `client-registration` service consumes the public keys referenced by `clients.yml` and reconciles the Flower SuperLink registration state. Existing known registrations are preserved; stale or changed identities are reconciled using the persistent registration manifest.

For local development on one machine, the server-side public keys generated for the clients are available in the same repository directory. For separate physical client hosts, copy **only** each client's `.pub` file to the server authorization directory.

### Start training

Once the required clients are online:

```bash
docker compose -f docker-compose.server.yml up trainer
```

The trainer always uses Flower's `production-deployment` profile.

## 5. Generated Compose files

The framework generates only two deployment files:

```text
docker-compose.server.yml
docker-compose.client.yml
```


## 6. Production deployment

Production deployment follows the same sequence as local distributed development, but the starter credentials must be replaced with valid federation-approved credentials.

### 6.1 Server credentials

The server requires:

```text
certificates/prod/tls/
├── ca.crt
├── superlink.crt
└── superlink.key
```

The CA must trust the SuperLink certificate. The SuperLink certificate SAN must contain the exact DNS name or IP used by `SUPERLINK_HOST`.

The SuperLink private key remains on the server and must never be copied to clients.

### 6.2 Client credentials

Each client host requires:

```text
certificates/prod/tls/ca.crt
certificates/prod/auth/<client-id>
```

The client authentication private key must remain on that client host. The matching public key is installed on the server and referenced by `clients.yml`.

A client must never receive another client's private key.

### 6.3 Replacing starter credentials

Starter credentials are created only when required material is missing. Existing files are preserved so that running setup does not silently overwrite production credentials.

Before production startup:

1. replace the starter CA/certificate/key with credentials issued through the approved federation PKI process;
2. install the federation CA on every client host;
3. install each client's approved SuperNode authentication private key only on its assigned client host;
4. install the corresponding public keys on the server;
5. update `clients.yml` if the public-key paths differ;
6. regenerate the server/client Compose files;
7. run host preparation and validation;
8. confirm that no starter credentials remain in the production deployment.

The setup script does not determine whether a credential is organizationally approved; that remains an operational/security responsibility.

## 7. Production startup sequence

On the server:

```bash
./setup.sh
```

Select `Prepare host`, then `Start server infrastructure`.


On each client host:

```bash
./setup.sh
```

Select `Prepare host`, then `Start client infrastructure`**`.

After the required clients are online, on the server:

```bash
docker compose -f docker-compose.server.yml up trainer
```

## 8. Network requirements

The production network must permit:

| Source | Destination | Port |
|---|---|---:|
| Client SuperNode | Server SuperLink | TCP 9092 |
| Server registration service | Server SuperLink | TCP 9093 |
| Server trainer | Server SuperLink | TCP 9093 |
| Server ServerApp | Server SuperLink | TCP 9091, Docker-internal |
| Client ClientApp | Local SuperNode | TCP 9094, Docker-internal |

The client hosts do not need access to server port 9091.

The server firewall should restrict Fleet API access to the participating client hosts where practical.

## 9. Security invariants

The deployment must maintain these rules:

- TLS is always enabled for the distributed SuperLink connections.
- SuperNode authentication is always enabled.
- The server never exposes its SuperLink private key to clients.
- A client receives only its own authentication private key.
- The server stores only client public authorization keys.
- SuperLink state is persisted outside the container.
- `--insecure` must not be used for the SuperLink or SuperNode network connections.
- Physical clients use a LAN DNS name or IP, never the Docker-only `superlink` hostname.
- Starter credentials are development/test credentials and must not be treated as production PKI.

## 10. Validation and troubleshooting

Show the resolved configuration with:

```bash
./setup.sh
```

Select `Show configuration`**`.

If a client cannot connect, check in this order:

1. `SUPERLINK_HOST` resolves/reaches the server from the client host.
2. TCP 9092 is permitted by the server firewall/network.
3. The client trusts the server `ca.crt`.
4. The SuperLink certificate SAN matches `SUPERLINK_HOST`.
5. The client's private authentication key matches the public key registered on the server.
6. The client ID exists in `clients.yml`.
7. The registration service has reconciled the client identity.
8. The server SuperLink state directory is persistent and writable.
9. The server and clients use the same repository revision and Flower version.

## 11. Testing

Run the automated test suite from the project environment with:

```bash
./setup.sh
```

Select `Run tests`**`.

The tests validate deployment configuration, secure Compose generation, DataContract behavior, and application validation.

The recommended federation integration test is the same multi-terminal procedure described in **Local distributed development**. This ensures local testing exercises the real server/client topology instead of an all-in-one simulation.
