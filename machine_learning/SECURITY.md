# Security Architecture and Policy

This document defines the security model, trust boundaries, credential policy, authentication controls, and production security requirements for the federated learning framework. It is the security specification; operational commands and deployment procedures belong in `DEPLOYMENT.md`.

## 1. Security objectives

The distributed federation must provide:

1. **Authenticated server endpoint** — SuperNodes verify the intended SuperLink.
2. **Encrypted federated transport** — SuperNode ↔ SuperLink Fleet communication uses TLS.
3. **Authenticated SuperNodes** — only registered SuperNode identities may participate.
4. **Credential isolation** — each client host receives only the credentials required for its own identity.
5. **Server private-key protection** — the SuperLink private key remains on the server.
6. **Persistent authorization state** — registered identities survive SuperLink container recreation when state is retained.
7. **No insecure downgrade** — distributed Fleet configuration must reject Flower's `--insecure` option.
8. **Trust-boundary separation** — internal Runtime/AppIO paths are distinct from Fleet and Control APIs.

These controls do not imply that the entire application is hardened against every possible threat.

## 2. One deployment profile

The framework has one Flower deployment profile:

```dotenv
DEPLOYMENT_PROFILE=production
```

The profile means the **secure distributed deployment configuration**. It is used for both local distributed development and production.

The credential source differs:

| Use | Deployment topology | Credential source |
|---|---|---|
| Local distributed development | Secure distributed | Generated starter credentials |
| Production | Secure distributed | Federation-approved credentials |

There is no insecure development profile and no `local-deployment` Flower profile.

## 3. Trust boundaries

```text
                         SERVER HOST
                 +-----------------------+
                 |       SuperLink       |
                 | Fleet API   :9092     |
                 | Control API :9093      |
                 | Runtime     :9091     |
                 +-----------+-----------+
                             |
                    TLS + authentication
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
          CLIENT 1       CLIENT 2       CLIENT 3
          SuperNode      SuperNode      SuperNode
          ClientApp      ClientApp      ClientApp
```

The external client/server trust boundary is the Fleet API on TCP 9092. The ClientApp ↔ SuperNode and ServerApp ↔ SuperLink Runtime/AppIO connections are local Docker paths and are not substitutes for Fleet security.

## 4. Flower network channels

| Connection | Port | Security | Purpose |
|---|---:|---|---|
| ServerApp/SuperExec → SuperLink | 9091 | Internal/plaintext | Server Runtime/AppIO |
| SuperNode → SuperLink | 9092 | **TLS + SuperNode authentication** | Federated communication |
| Trainer/registration → SuperLink | 9093 | **TLS** | Control operations |
| ClientApp/SuperExec → SuperNode | 9094 | Internal/plaintext | Client Runtime/AppIO |

A physical client must use the server's LAN DNS name or IP. The Docker-only hostname `superlink` must not be used as a physical federation endpoint.

The current `--insecure` options on the ServerApp and ClientApp Runtime/AppIO services are internal-path settings and do not disable Fleet TLS/authentication. Runtime/AppIO TLS remains future hardening.

## 5. TLS architecture

The SuperLink holds:

```text
/etc/flower/tls/
├── ca.crt
├── superlink.crt
└── superlink.key
```

SuperNodes receive only `ca.crt`. The SuperLink certificate must contain a Subject Alternative Name matching the DNS name or IP used by the connecting SuperNode and control client.

For example, if clients connect to `192.168.0.172:9092`, the certificate must contain `IP:192.168.0.172` in its SAN.

The CA private signing key must never be stored in the repository or distributed to clients. The repository-generated starter CA is for controlled development/testing only. Production certificates must come from the organization's approved PKI process.

## 6. SuperNode authentication

TLS establishes trust in the SuperLink endpoint. SuperNode authentication establishes authorization for the connecting client identity. Both controls apply to the production Fleet connection.

Each client has a unique ECDSA P-384 key pair in OpenSSH format:

```text
client-1 private key  → retained only by client-1
client-1 public key   → registered/authorized on the server
```

The SuperLink enables authentication with:

```text
--enable-supernode-auth
```

A client uses its own identity with:

```text
--auth-supernode-private-key /etc/flower/auth/<client-id>
```

A client host must never receive another client's private key, the SuperLink private key, or the CA private signing key.

## 7. Public-key authorization

Key generation alone does not authorize a client. The lifecycle is:

```text
Generate key pair
      |
      +---- private key stays on client
      |
      +---- public key --> server authorization inventory
                              |
                              v
                         Flower registry
                              |
                              v
                         SuperNode access
```

`clients.yml` is the server-side inventory of authorized public keys:

```yaml
clients:
  - id: client-1
    public_key: ./certificates/auth/client-1.pub
```

The registration service needs public keys, not client private keys.

## 8. Persistent authorization state

SuperLink state is stored outside the container using a database equivalent to:

```text
--database /var/lib/flower/superlink.db
```

The host state directory must survive ordinary container recreation. Deleting it is an administrative/security operation because it can alter the federation's authorization state.

The registration workflow also maintains a server-side manifest so that existing Flower registrations can be reconciled without guessing identities from node IDs.

## 9. Credential handling

Never commit:

- populated `.env` files;
- SuperLink private keys;
- SuperNode private authentication keys;
- CA private signing keys;
- runtime state databases;
- other deployment-specific secrets.

The generated Compose files mount credential material read-only where the service only consumes it.

Credential scope follows least privilege:

| Service | Credential scope |
|---|---|
| SuperLink | CA, SuperLink certificate/private key, persistent state |
| SuperNode | Federation CA + its own private authentication key |
| Registration service | Federation CA + public client keys |
| ClientApp | No SuperNode private key |
| ServerApp | No client private keys |

## 10. Starter credentials and production credentials

The setup helper may generate starter TLS and SuperNode authentication credentials when the expected files do not exist. These credentials make local distributed development reproducible but are **not production PKI credentials**.

Setup must never silently overwrite an existing complete credential set. Before production deployment, replace starter material with credentials issued/approved by the federation's credential-management and PKI process.

Production preparation should verify:

- CA, certificate, and key files exist and are parseable;
- the SuperLink certificate chains to the installed CA;
- the certificate SAN matches `SUPERLINK_HOST`;
- each configured client has a matching public authentication key;
- each client host has only its assigned private key;
- the SuperLink private key is absent from client hosts.

Organizational approval of a credential cannot be determined automatically by this repository and remains an operational responsibility.

## 11. Key rotation and revocation

If a client private key is compromised:

1. Stop the affected SuperNode.
2. Revoke/unregister the old identity through the SuperLink administration process.
3. Generate or provision a new approved key pair.
4. Install the new private key only on the affected client host.
5. Register the new public key.
6. Verify the new identity before resuming federation.

Do not overwrite a private key while leaving the old public key authorized.

Certificate rotation must preserve the trusted CA chain and the SAN identity required by the deployed endpoint.

## 12. Network exposure policy

Only externally required APIs should cross the physical host boundary.

The intended external client path is:

```text
Client SuperNode  ── TCP 9092 ──>  Server SuperLink Fleet API
```

Port 9091 is a server-side Runtime/AppIO path and should not be exposed to clients. Port 9094 is local to each client Compose deployment. Port 9093 is an administrative/control interface and should be restricted to the server-side components that require it.

Host firewalls should restrict Fleet access to participating client hosts where practical.

## 13. Failure behavior

Authentication failures must fail closed.

An unregistered SuperNode or a SuperNode using the wrong private key must not participate. TLS verification failure must not be bypassed by enabling `--insecure`.

Expected behavior:

```text
TLS validation
     |
     +---- fail ----> no federation access
     |
     v
SuperNode authentication
     |
     +---- fail ----> no federation access
     |
     v
Authorized federation participation
```

## 14. Runtime/AppIO security boundary

Current status:

```text
Fleet 9092       TLS + SuperNode authentication       IMPLEMENTED
Control 9093     TLS                                  IMPLEMENTED
Runtime 9091     Internal/plaintext                   FUTURE HARDENING
Runtime 9094     Internal/plaintext                   FUTURE HARDENING
```

Runtime/AppIO TLS requires separate certificate, key, trust, and SAN handling. The SuperLink private key must not be reused as a shared credential for those services.

## 15. Security verification

A production security verification should demonstrate:

1. An authorized SuperNode establishes the TLS-protected Fleet connection.
2. An unregistered identity is rejected.
3. A wrong private key is rejected.
4. A certificate/SAN mismatch causes TLS failure.
5. Secure production commands reject `--insecure` on Fleet/SuperNode paths.
6. Client hosts contain only their assigned private identity.
7. The SuperLink private key is absent from clients.
8. Authorization survives SuperLink recreation when persistent state is retained.
9. ClientApp and ServerApp containers do not receive unnecessary private authentication credentials.

The concrete procedure is maintained in `DEPLOYMENT.md`.

## 16. Security limitations and future hardening

The current TLS/authentication layer does not by itself provide:

- secure aggregation;
- differential privacy;
- protection against malicious or poisoned model updates;
- end-to-end Runtime/AppIO encryption;
- comprehensive centralized audit logging;
- production PKI lifecycle automation;
- full network segmentation or firewall automation.

These remain separate engineering and operational concerns.
