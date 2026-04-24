# 1Password CTF — Engagement Log

## Step 1.1 — Infrastructure Scan

**Date:** 2026-04-23
**Target:** bugbounty-ctf.1password.com
**Scan:** Full port scan (1-65535), 500 threads

### Open Ports

| Port | Protocol | Service | Banner |
|------|----------|---------|--------|
| 80 | tcp | HTTP | `awselb/2.0` — returns 403 Forbidden |
| 443 | tcp | HTTPS | `awselb/2.0` — returns 400 (plain HTTP probe to TLS port) |

All other ports (1-65535) closed or filtered.

### Infrastructure

- Behind **AWS Elastic Load Balancer** (`awselb/2.0`)
- No direct access to application servers
- Port 80 returns `403 Forbidden` with body "HTTP Forbidden" — HTTP access blocked, forces HTTPS
- OS fingerprint: Unknown (ELB masks backend)

### False Positives

The NVD scanner flagged two CVEs by generic string match against "HTTP" — neither applies:
- CVE-2017-9788 (Apache httpd mod_http2) — server is awselb, not Apache
- CVE-2017-5638 (Apache Struts RCE) — no Struts in evidence

### Assessment

Minimal attack surface. Only standard web ports exposed, both behind AWS ELB.
No SSH, no database ports, no admin interfaces, no non-standard services.
The engagement proceeds entirely through the HTTPS endpoint on port 443.


## Step 1.2 — TLS Configuration Audit

**Date:** 2026-04-23

### Protocol Versions

| Version | Accepted | Cipher Negotiated | Bits |
|---------|----------|-------------------|------|
| TLS 1.3 | Yes | TLS_AES_128_GCM_SHA256 | 128 |
| TLS 1.2 | Yes | ECDHE-RSA-AES128-GCM-SHA256 | 128 |
| TLS 1.1 | **No** | — | — |
| TLS 1.0 | **No** | — | — |

No downgrade to TLS 1.1 or 1.0. POODLE, BEAST, and legacy protocol attacks are
not applicable.

### Cipher Suites Accepted

**TLS 1.3** (1 suite):
- `TLS_AES_128_GCM_SHA256` (128-bit)

**TLS 1.2** (4 suites):
- `ECDHE-RSA-AES256-GCM-SHA384` (256-bit)
- `ECDHE-RSA-AES128-GCM-SHA256` (128-bit)
- `ECDHE-RSA-AES256-SHA384` (256-bit)
- `ECDHE-RSA-AES128-SHA256` (128-bit)

**Weak ciphers:** None. No RC4, DES, 3DES, export-grade, or NULL suites. All
suites use ECDHE for forward secrecy.

**Note:** TLS 1.3 only negotiated AES-128-GCM, not AES-256-GCM or
CHACHA20-POLY1305. This is likely an AWS ELB default preference — not a
vulnerability, but worth noting that the server prefers 128-bit over 256-bit.

### Certificate

| Field | Value |
|-------|-------|
| Subject | `CN=1password.com` |
| Issuer | `CN=Amazon RSA 2048 M01, O=Amazon, C=US` |
| Key | RSA 2048-bit |
| Signature | SHA-256 with RSA |
| Valid from | 2026-01-22 |
| Valid until | 2027-02-20 |
| Days remaining | 302 |
| SANs | `1password.com`, `*.1password.com` |
| OCSP | `http://ocsp.r2m01.amazontrust.com` |
| Version | v3 |

Wildcard cert covering all `*.1password.com` subdomains. Amazon-issued (ACM).
RSA 2048 is the minimum recommended key size — adequate but not exceptional.
No ECC key.

### Security Headers

The root path (`/`) returns `403` from the ELB with minimal headers:
- `x-content-type-options: nosniff` — present
- `Strict-Transport-Security` — **absent** on the 403 response
- `Content-Security-Policy` — **absent**
- `X-Frame-Options` — **absent**
- `Server` header — **absent** (good, no server fingerprinting)

The missing HSTS on the 403 is likely because the ELB default page doesn't
set it. The actual application pages (login, vault UI) may set HSTS separately.
This should be verified in Step 1.3 (Web Client Extraction).

### Assessment

TLS configuration is solid:
- No protocol downgrade path (TLS 1.1/1.0 rejected)
- All cipher suites use AEAD modes with ECDHE forward secrecy
- No weak or deprecated ciphers
- Certificate is valid, properly chained, with appropriate SANs

Minor observations (not vulnerabilities):
- RSA-2048 key (minimum recommended; EC P-256 or RSA-4096 would be stronger)
- TLS 1.3 prefers AES-128-GCM over AES-256-GCM
- HSTS not observed on ELB 403 page — **confirmed present on application pages**
  (see Step 1.3)


## Step 1.3 — Web Client Extraction

**Date:** 2026-04-23

### Application Structure

The root URL (`/`) serves the full SPA. All paths (`/signin`, `/sign-in`,
`/login`, `/app`) return the same shell HTML — client-side routing. The `/app`
path returns a slightly different CSP (more restrictive).

- **Build version:** `data-version="2248"`
- **Git revision:** `data-gitrev="33a8e241e543"`
- **Build time:** 23 Apr 26 18:49 +0000 (same day as our scan)
- **Environment:** `prd` (production)
- **Canonical URL:** `https://my.1password.com/`
- **Sibling domains:** `1password.ca`, `1password.eu`, `ent.1password.com`

### JavaScript Bundles

All served from `https://app.1password.com/` with SRI integrity hashes:

| Bundle | Hash (truncated) | Purpose |
|--------|-------------------|---------|
| `runtime-62c8ad17.min.js` | `sha384-lnpYOr...` | Webpack runtime |
| `vendor-1password-383fec46.min.js` | `sha384-ps/sIb...` | 1Password core library |
| `vendor-other-8afa0afd.min.js` | `sha384-yTVzGZ...` | Third-party deps |
| `vendor-react-7f2b22fd.min.js` | `sha384-AxAeyL...` | React framework |
| `vendor-lodash-11dceb72.min.js` | `sha384-/jCcn7...` | Lodash utilities |
| `webapi-d3ad37f2.min.js` | `sha384-0oSoS6...` | Web API client |
| `vendor-moment-a350876a.min.js` | `sha384-bgHnUo...` | Date/time library |
| `app-4b7678e0.min.js` | `sha384-PdqkKN...` | Main application |
| `sk-2c17b526.min.js` | `sha384-9UxhaJ...` | Secret Key retrieval (fallback) |

All scripts use `crossorigin="anonymous"` and SRI hashes — tampering with the
CDN content would be detected by the browser.

### WebAssembly Security

The client ships WASM modules (likely the crypto core) with a **hash whitelist**:

```
trustedWasmHashes = [
    'k6RLu5bHUSGOTADUeeTBQ1gSKjiazKFiBbHk0NxflHY=',
    'L7kNpxXKV0P6GuAmJUXBXt6yaNJLdHqWzXzGFEjIYXQ=',
    'GVnMETAEUL/cu/uTpjD6w6kwDLUYqiEQ7fBsUcd+QJw=',
    '+yHBrSgjtws1YuUDyoaT3KkY0eOi0gVCBOZsGNPJcOs=',
    'I+k/SNmZg4ElHUSaENw7grySgWIki/yjg62WZcsxXy8=',
    'WwqUPAGJ2F3JdfFPHqHJpPrmVI5xmLlfIJadWXKRQR8='
]
```

Every WASM module is SHA-256 hashed before loading and compared against this
list. `WebAssembly.compile`, `instantiate`, `validate`, and
`compileStreaming` are all monkey-patched to enforce this check. The non-async
`Module` constructor is blocked entirely.

This is a defense against WASM substitution attacks — even with a MITM, an
attacker cannot inject a modified crypto module without matching one of these
hashes. **This significantly raises the bar for client-side attacks.**

WASM base URL: `https://app.1password.com/wasm/`

### Security Headers (Application Pages)

All security headers are present and well-configured on the application pages:

| Header | Value |
|--------|-------|
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains; preload` |
| `Content-Security-Policy` | Strict — see below |
| `X-Frame-Options` | `DENY` |
| `X-Content-Type-Options` | `nosniff` |
| `Referrer-Policy` | `no-referrer` |
| `Cross-Origin-Opener-Policy` | `restrict-properties` |
| `Permissions-Policy` | `interest-cohort=()` |
| `Cache-Control` | `max-age=60, no-cache, no-store` |
| CSP Reporting | `report-to csp-endpoint` -> `https://csp.1passwordservices.com/report` |

### Content Security Policy (Parsed)

```
default-src:       'none'
script-src:        https://app.1password.com 'wasm-unsafe-eval' + 2 inline hashes
style-src:         https://app.1password.com + 1 inline hash
connect-src:       'self' blob: https://app.1password.com wss://b5n.1password.com
                   https://*.1password.com https://*.1password.ca https://*.1password.eu
                   https://*.ent.1password.com https://f.1passwordusercontent.com
                   https://a.1passwordusercontent.com https://watchtower.1password.com
                   https://api.pwnedpasswords.com + Firebase, Sentry, telemetry
font-src:          https://app.1password.com
img-src:           data: blob: https://app.1password.com + avatar/cache CDNs
child-src/frame-src: 'self' + Duo Security, billing, survey, email providers
worker-src:        'self'
form-action:       https://app.kolide.com/ https://app.trelica.com/
frame-ancestors:   https://*.1password.com
upgrade-insecure-requests
```

**CSP Analysis:**
- `default-src 'none'` — strict baseline, everything must be explicitly allowed
- `script-src` — **no `unsafe-inline` or `unsafe-eval`** — only hashed inlines
  and `https://app.1password.com`. `wasm-unsafe-eval` is required for WASM
  execution but is mitigated by the WASM hash whitelist
- `connect-src` — allows WebSocket to `wss://b5n.1password.com` (push notifications?)
  and HTTPS to various 1Password service domains
- `frame-ancestors: https://*.1password.com` — prevents clickjacking from
  non-1password origins
- CSP violation reporting is active — any injection attempt would be reported

**XSS attack surface is very limited.** No `unsafe-inline`, no `unsafe-eval`,
SRI on all scripts, WASM hash whitelist, strict frame-ancestors.

### Exposed Configuration Data

The HTML `<head>` tag contains `data-*` attributes with configuration:

**Potentially interesting for the engagement:**
- `data-brex-client-id`: `bri_b2df18d65bc82a948573537157eceb07`
- `data-brex-auth`: `CLIENT_SECRET` (literal string, not an actual secret)
- `data-fcm-api-key`: `AIzaSyCs8WNa10YE5AVyfL33RBHBKQdYZMw7OB0` (Firebase Cloud Messaging)
- `data-fcm-project-id`: `b5-notification-prd`
- `data-sentry-dsn`: `https://6342e577bc314e54ab2c5650a4c5be8f:f7b7d11056d84dd0b09e9a9ca31a72e8@web-ui-sentry.1passwordservices.com/...`
- `data-slack-client-id`: `36986904051.273534103040`
- `data-stripe-key`: `pk_live_F59R8NjiAi5Eu7MJcnHmdNjj`
- `data-fastmail-client-id`: `35c941ae`
- `data-snowplow-url`: `https://telemetry.1passwordservices.com` (analytics)
- `data-webpack-public-path`: `https://app.1password.com/` (CDN origin)

The page includes `data-bug-researcher-notes` that explicitly states: "All keys
below are intended to be exposed publicly, and are therefore not vulnerable."

### Assessment

The web client is well-hardened:
- SRI on all scripts prevents CDN tampering
- WASM hash whitelist prevents crypto module substitution
- Strict CSP blocks most XSS vectors
- HSTS with preload prevents SSL stripping
- `X-Frame-Options: DENY` prevents clickjacking
- CSP violation reporting is active

The main avenue for client-side attacks would be:
1. Finding an XSS that works within the CSP constraints (very difficult)
2. Compromising `app.1password.com` CDN itself (the only allowed script source)
3. Exploiting `wasm-unsafe-eval` if a WASM module can be substituted (blocked by
   hash whitelist, but worth investigating the validation code path)

The `vendor-1password` and `webapi` bundles are the highest-value targets for
reverse engineering — they contain the SRP client, key derivation, and vault
encryption logic.


## Step 1.4 — API Enumeration

**Date:** 2026-04-23

### CORS Configuration

`OPTIONS /api/v1/auth` returns:
- `access-control-allow-origin: https://bugbounty-ctf.1password.com` (strict, not `*`)
- `access-control-allow-credentials: true`
- `access-control-allow-headers: X-AgileBits-Client, X-AgileBits-MAC, Cache-Control, X-AgileBits-Session-ID, Content-Type, OP-User-Agent, ChannelJoinAuth`
- `access-control-allow-methods: GET, POST, PUT, PATCH, DELETE`

Notable custom headers: `X-AgileBits-Client`, `X-AgileBits-MAC`,
`X-AgileBits-Session-ID` — likely required for authenticated requests.
The MAC header suggests request signing.

### Auth Endpoints

| Endpoint | Method | Status | Response |
|----------|--------|--------|----------|
| `/api/v1/auth` | POST | 401 | `{}` (empty, no differentiation by email) |
| `/api/v2/auth` | POST | 401 | `{}` |
| `/api/v2/auth/complete` | POST | 401 | `{}` |
| `/api/v2/auth/confirm-key` | POST | 401 | `{}` |
| `/api/v2/auth/methods` | POST | **200** | `{"authMethods":[{"type":"PASSWORD+SK"}],...}` |
| `/api/v1/auth/verify` | POST | 401 | `{}` |
| `/api/v1/auth/mfa` | POST | 401 | `{}` |
| `/api/v3/auth` | POST | 404 | No v3 API |

The auth init endpoint returns identical `401 {}` for all email addresses
including empty string — **no username enumeration** via this path.

### Key Finding: `/api/v2/auth/methods`

This endpoint returns 200 for any request and confirms:
```json
{"authMethods":[{"type":"PASSWORD+SK"}],"signInAddress":"https://bugbounty-ctf.1password.com"}
```

- Auth method is `PASSWORD+SK` (password + Secret Key, i.e., 2SKD)
- Returns the same response for all emails including empty/nonexistent
- Returns 400 only for malformed email strings (e.g., `"not-an-email"`)
- **No SSO** — pure password + Secret Key auth only
- **No email enumeration** possible through this endpoint

### Endpoint Map (from JS Bundle)

The `webapi` bundle (934 KB) contains ~200 API endpoint paths. Key categories:

**Auth flow (v2):**
- `/api/v2/auth` — SRP init
- `/api/v2/auth/complete` — SRP verify / session creation
- `/api/v2/auth/confirm-key` — Secret Key confirmation
- `/api/v2/auth/methods` — query auth methods (public)
- `/api/v2/auth/webauthn/register` — WebAuthn registration
- `/api/v2/auth/webauthn/register/challenge` — WebAuthn challenge
- `/api/v2/auth/sso/reconnect` — SSO reconnection

**Recovery (v2) — high-value attack surface:**
- `/api/v2/recovery-keys/session/new` — start recovery session
- `/api/v2/recovery-keys/session/auth/cv1/start` — recovery auth start
- `/api/v2/recovery-keys/session/auth/cv1/confirm` — recovery auth confirm
- `/api/v2/recovery-keys/session/complete` — complete recovery
- `/api/v2/recovery-keys/session/identity-verification/email/start` — email verification
- `/api/v2/recovery-keys/session/identity-verification/email/submit` — submit verification
- `/api/v2/recovery-keys/session/material` — recovery key material
- `/api/v2/recovery-keys/session/status` — session status
- `/api/v2/recovery-keys/policies` — recovery policies (returns 401)
- `/api/v2/recovery-keys/keys` — recovery keys (returns 401)
- `/api/v2/recovery-keys/attempts` — recovery attempts (returns 401)

**Account/keyset management:**
- `/api/v2/account/keysets` — account keysets (returns 401)
- `/api/v1/account` — account info (returns 401)
- `/api/v1/device` — device registration (returns 401)
- `/api/v1/session/signout` — session termination
- `/api/v1/session/touch` — session keepalive
- `/api/v2/session-restore/*` — session restore flow (save-key, restore-key, destroy-key)

**Vault operations:**
- `/api/v2/vault` — vault access
- `/api/v2/mycelium/u` / `/api/v2/mycelium/v` — unknown (Mycelium?)
- `/api/v1/vault/personal` — personal vault
- `/api/v1/vault/everyone` — shared vault
- `/api/v1/vault/managed` — managed vault
- `/api/v1/vault/account-transfer` — vault transfer

**Other interesting:**
- `/api/v1/confidential-computing/session` — confidential computing
- `/api/v1/signinattempts` / `/api/v2/signinattempts` — sign-in attempt logs
- `/api/v1/monitoring/status` — monitoring
- `/api/v2/perftrace` / `/api/v2/preauth-perftrace` — performance tracing
- `/api/v1/oidc/token` — OIDC token endpoint

### Error Behavior

All authenticated endpoints return `401 {}` (empty JSON body) — the server
leaks no information about why the request failed. No differentiated error
messages, no descriptive error codes.

Signup endpoints (`/api/v1/signup`, `/api/v2/signup`) return `400 {}` for all
payloads — signup may be disabled on the CTF instance.

### Rate Limiting

5 rapid sequential requests to `/api/v1/auth` all returned `401` with no
throttling or blocking. No `Retry-After` header. No CAPTCHA challenge.
**Rate limiting may be absent or has a high threshold.**

### Assessment

The API surface is large (~200 endpoints) but consistently requires
authentication. Key observations:

1. **No username/email enumeration** — all auth endpoints return identical
   responses regardless of email
2. **Recovery key flow is extensive** — 10+ endpoints for account recovery.
   This is the white paper's Appendix A.4 weakness. Worth deep investigation
   in Phase 3.
3. **Custom request signing** — `X-AgileBits-MAC` header suggests HMAC-based
   request authentication. Need to understand this from the JS bundle.
4. **Session restore flow** — save/restore/destroy key endpoints could be
   a secondary attack surface for session hijacking.
5. **No rate limiting observed** — brute force may be feasible if the auth
   protocol allows it (2SKD makes this moot for password attacks, but
   session/token brute force could be viable).
6. **v2 auth flow** — the client uses v2 (`/api/v2/auth` -> `/api/v2/auth/complete`)
   rather than v1. Both respond similarly.


## Step 1.5 — Public Source Analysis

**Date:** 2026-04-23

### 1Password Public Repositories

93 public repos on GitHub under `github.com/1Password`. Relevant repos:

**Highest value:**

| Repo | Language | Description |
|------|----------|-------------|
| `1Password/srp` | Go | **SRP-6a implementation used by 1Password Teams** (389 stars) |
| `burp-1password-session-analyzer` | Java | **Burp plugin for analyzing encrypted 1Password sessions** (79 stars) |
| `passkey-rs` | Rust | WebAuthn authenticator framework |
| `curve25519-dalek` | Rust | Fork with specific bug fix |

### SRP Library Analysis (`1Password/srp`)

**Files:** 13 Go source files, ~70KB total. Full SRP-6a implementation with both
standard (RFC 5054) and non-standard (1Password legacy) modes.

**Key classes:**
- `SRP` struct — main client/server object
- `Group` — Diffie-Hellman group parameters
- `Hash` — configurable hash (SHA-256 default)

#### Critical Validation: `IsPublicValid()` (srp.go:208)

```go
func (s *SRP) IsPublicValid(AorB *big.Int) bool {
    if s.group.Reduce(AorB).Cmp(bigOne) == 0 {
        return false  // Rejects A % N == 1
    }
    if s.group.IsZero(AorB) {
        return false  // Rejects A == 0
    }
    return true
}
```

**Assessment:** This validates A != 0 and A % N != 1, but **does NOT check
A % N != 0** directly. The `Reduce` call computes `A mod N`. If `A = N`, then
`Reduce(A) = 0`, which is caught by `IsZero`. If `A = 2N`, then `Reduce(A) = 0`,
also caught. But the check for `Cmp(bigOne)` only catches `A % N == 1`, not
`A % N == 0` when A > 0.

Wait — re-reading: `IsZero` checks if the value is zero. `Reduce(A)` gives
`A mod N`. So:
- A=0: `IsZero(0)` = true -> rejected
- A=N: `Reduce(N) = 0`, `IsZero(0)` = true -> rejected
- A=2N: `Reduce(2N) = 0`, `IsZero(0)` = true -> rejected
- A=kN: same, all rejected

**The zero-key attack is properly mitigated in this library.** The library also
rejects A=1 (which would make the session key deterministic but not trivially
zero). Additional safety: `SetOthersPublic()` calls `IsPublicValid()` and sets
`badState=true` on failure, preventing any further key computation.

#### Non-Standard u Calculation

The library has a documented bug:
```go
// BUG(jpg): Calculation of u does not use RFC 5054 compatible padding/hashing
```

The non-standard mode (`calculateUNonStd`) concatenates hex strings of A and B
with leading zeros stripped, then hashes. This differs from RFC 5054 which
requires fixed-width padding. The standard mode (`calculateUStd`) uses proper
padding. **The web client likely uses the standard mode** (`NewClientStd`), but
this should be verified.

#### Other Observations

- SHA-256 is hardcoded as the default hash
- Ephemeral secret minimum size: 32 bytes (per RFC 5054)
- `u == 0` is explicitly rejected (would make session key independent of password)
- Server-side key computation: `S = (A * v^u) ^ b mod N`
- Client-side key computation: `S = (B - k*g^x) ^ (a + u*x) mod N`

### Burp Plugin Insight

The `burp-1password-session-analyzer` README reveals critical architecture:

> "We require every request and response that are specific to a 1Password account
> to be protected by the account's master password and secret key, which means
> every bit of data that gets sent is encrypted, and every request is authenticated."

This confirms:
1. **All API payloads are encrypted** — not just auth, ALL requests/responses
2. **Every request is MAC'd** — explains the `X-AgileBits-MAC` header from Step 1.4
3. **Standard web fuzzing tools don't work** — you can't tamper with requests
   without the session key
4. The Burp plugin requires a valid session key to decrypt/re-encrypt payloads

This means:
- IDOR/parameter tampering is not possible without first obtaining a valid session
- API fuzzing requires understanding the encryption layer
- The `X-AgileBits-Session-ID` + `X-AgileBits-MAC` headers are integral to the
  protocol, not optional

### Assessment

The SRP library is well-implemented:
- Zero-key attacks (A=0, A=N, A=kN) are properly rejected
- The library is well-tested (20KB of tests)
- SHA-256 is used throughout
- Session key derivation follows standard SRP-6a

The main attack surface from source analysis:
1. **Non-standard u calculation** — if the server uses the legacy mode, the
   different padding could theoretically be exploitable, though this is unlikely
2. **All-encrypted API protocol** — makes server-side testing much harder than
   anticipated. We need the session key to even send valid requests
3. **Burp plugin exists** — we should use this for any authenticated testing


## Step 1.6 — CVE / Exploit Search

**Date:** 2026-04-23

**Source:** NVD CVE List V5 database (346,306 CVEs loaded into local SQLite),
cross-referenced with agent-gathered research from Exploit-DB, academic papers,
and security advisories.

### 1Password-Specific CVEs (13 total)

| CVE | CVSS | Product | Relevance |
|-----|------|---------|-----------|
| **CVE-2022-32550** | — | All 1Password apps | **SRP connection validation deviation** — server impersonation possible in specific circumstances. The only CVE targeting 1Password's SRP implementation. Patched. |
| **CVE-2020-10256** | 9.8 | CLI/SCIM Bridge (beta) | **Insecure PRNG for encryption keys** — brute-forceable key generation. Beta-only, not main apps. Patched. |
| **CVE-2024-42219** | 7.8 | 1Password 8 macOS | **XPC IPC validation bypass** — local attacker exfiltrates vault items + SRP-x via impersonating browser extension. Patched 8.10.36. |
| **CVE-2024-42218** | 4.7 | 1Password 8 macOS | **Downgrade attack** — local attacker uses old app version to bypass macOS security. Patched 8.10.38. |
| **CVE-2022-29868** | 5.5 | 1Password 7 macOS | **Process validation bypass** — local exfiltration of secrets including "derived values used for signing in." Patched 7.9.3. |
| **CVE-2021-41795** | 6.5 | Safari extension | **Authorization bypass** — malicious web page reads fillable vault items silently. Patched 7.8.7. |
| **CVE-2021-36758** | 5.4 | Connect server | **Privilege escalation** via improperly scoped access tokens. Patched 1.2. |
| **CVE-2021-26905** | 6.5 | SCIM Bridge | **TLS private key disclosure** via log file access. Patched 1.6.2. |
| **CVE-2020-18173** | 7.8 | 1Password 7 Windows | **DLL injection** — local arbitrary code execution. |
| **CVE-2018-19863** | 5.5 | 1Password 7 macOS | **Credential logging** — Safari→1Password data logged locally. Patched. |
| **CVE-2018-13042** | 5.9 | 1Password 6 Android | **DoS** via exported activities. Not relevant to web. |
| **CVE-2014-3753** | 5.5 | 1Password Windows | **Security feature bypass.** Sparse details. |
| **CVE-2012-6369** | 4.3 | 1Password 3 desktop | **XSS** in troubleshooting report. Ancient, irrelevant. |

**Assessment:** No CVE has ever achieved remote vault content recovery against
1Password. All high-severity CVEs require local access (macOS IPC bypass). The
only SRP-related CVE (CVE-2022-32550) was a connection validation issue, not a
cryptographic break. The insecure PRNG (CVE-2020-10256) only affected beta CLI
tools.

### SRP Protocol CVEs

| CVE | CVSS | Description | Applies to 1Password? |
|-----|------|-------------|----------------------|
| **CVE-2009-4810** | 7.5 | Samhain SRP zero-value validation bypass (classic A=0) | **NO** — 1Password's library validates via `IsPublicValid()` |
| **CVE-2025-54885** | 6.9 | Thinbus JS SRP: 252 bits entropy instead of 2048 (function vs value bug) | **NO** — different library, JS-specific bug |
| **CVE-2026-3559** | 8.1 | Philips Hue: SRP static nonce, full auth bypass | **NO** — implementation bug, not protocol flaw |
| **CVE-2021-4286** | 2.6 | pysrp: timing leak in `calculate_x` | **POSSIBLY** — same attack class (timing) is relevant |

### SRP Academic Research

| Paper | Year | Finding | Relevance |
|-------|------|---------|-----------|
| **PARASITE** (CCS 2021) | 2021 | OpenSSL `BN_mod_exp` non-constant-time path leaks password info via cache timing. Single-trace attack. | **HIGH** — if 1Password's server uses affected OpenSSL version. The Go SRP library uses Go's `math/big`, not OpenSSL. |
| **Threat for SRP** (ACNS 2021) | 2021 | MitM can modify salt to derive new exponent, exploiting timing even with different client implementation | **MEDIUM** — requires MitM + timing vulnerability |
| **Just How Secure is SRP?** (ePrint 2025) | 2025 | SRP is probably NOT UC-secure; existing proof uses non-standard model | **LOW** — theoretical; game-based security still holds |
| **Small subgroup non-confinement** (Hao 2010) | 2010 | Information leakage from subgroup structure | **LOW** — mitigated by safe primes |

### PBKDF2 CVEs

| CVE | CVSS | Description | Relevance |
|-----|------|-------------|-----------|
| **CVE-2025-6545** | 9.1 | npm `pbkdf2` package: returns zero-filled buffers for non-normalized algorithm names | **CHECK** — if web client uses this polyfill instead of native WebCrypto |
| **CVE-2025-6547** | 9.1 | Same `pbkdf2` package: improper validation | Same as above |
| **CVE-2023-46233** | 9.1 | crypto-js: PBKDF2 defaults to SHA-1 with 1 iteration | **NO** — 1Password uses explicit SHA-256 + 100k+ iterations |
| **CVE-2023-46133** | 9.1 | CryptoES: same weak default as crypto-js | **NO** — same reason |
| **CVE-2025-11187** | — | OpenSSL PBMAC1: stack buffer overflow in PKCS#12 MAC verification | **NO** — different context (PKCS#12) |

**Key observation:** 1Password's default PBKDF2-HMAC-SHA256 iterations is
**650,000** (discovered in Step 3.9 — `DEFAULT_ITERATIONS=65e4`), which exceeds
OWASP's 2025 recommendation of 600,000. A secondary constant of 100,000 exists
for token-based derivation. The 128-bit Secret Key makes brute force infeasible
regardless.

### AES-GCM / Nonce CVEs

| CVE | CVSS | Description | Relevance |
|-----|------|-------------|-----------|
| **CVE-2026-5446** | 6.0 | wolfSSL ARIA-GCM: reuses identical 12-byte nonce for every record | **PATTERN** — demonstrates catastrophic nonce reuse |
| **CVE-2026-26014** | 5.9 | Pion DTLS: random nonce generation, birthday bound collision | **PATTERN** — random nonces hit collision at 2^32 messages |
| **CVE-2021-32791** | 5.9 | mod_auth_openidc: static IV for AES-GCM | **PATTERN** — static nonce = keystream recovery |
| **CVE-2025-61739** | 7.2 | Generic nonce reuse: replay attack or decryption | **PATTERN** |

**Assessment:** No AES-GCM CVE directly affects 1Password. The nonce reuse
pattern is the primary risk — must verify 1Password uses unique per-item nonces.
Birthday bound (2^32 messages per key) is unlikely to be reached in vault usage.

### WebCrypto CVEs

| CVE | CVSS | Description | Relevance |
|-----|------|-------------|-----------|
| **CVE-2016-5142** | 9.8 | Chrome WebCrypto use-after-free — RCE | **HISTORICAL** — fixed Chrome 52 (2016) |
| **CVE-2017-7822** | 5.3 | Firefox: AES-GCM accepts zero-length IV | **HISTORICAL** — fixed Firefox 56 (2017) |
| **CVE-2022-35255** | — | Node.js: weak randomness in WebCrypto keygen | **NO** — browser, not Node.js |
| **CVE-2018-5122** | — | Firefox: integer overflow in WebCrypto DoCrypt | **HISTORICAL** — fixed |

**Assessment:** All WebCrypto CVEs are historical and patched in modern browsers.
The browser's native crypto layer is the correct choice over JS polyfills.

### Indirect / Dependency CVEs

| CVE | CVSS | Description | Relevance |
|-----|------|-------------|-----------|
| **CVE-2023-4863** | 10.0 | libwebp heap buffer overflow (via Chromium/Electron) | 1Password patched in 8.10.15. RCE via crafted WebP image. |
| **CVE-2025-55305** | — | Electron ASAR integrity bypass (Trail of Bits) | 1Password patched in 8.11.8-40. Local backdoor via V8 snapshot. |

### Research Papers

| Paper | Finding | Relevance |
|-------|---------|-----------|
| **ETH Zurich / USI** (USENIX Sec 2026) | 2 attack scenarios under malicious-server model achieve full vault compromise | **HIGH** — but 1Password says these are documented in their white paper. The 2SKD Secret Key provides protection competitors lack. |
| **DOM-based extension clickjacking** (DEF CON 33, 2025) | Clickjacking attacks against browser extension autofill | **MEDIUM** — patched in extension 8.11.7. Not relevant to web vault. |

### Summary Assessment

1. **No remote vault content recovery has ever been demonstrated** against
   1Password via any CVE or published research
2. **SRP implementation is solid** — zero-key attacks properly mitigated,
   connection validation CVE (2022-32550) is patched
3. **PARASITE timing attack is the most credible SRP threat** — but requires
   non-constant-time big number operations, and Go's `math/big` (used by the
   SRP library) is not known to have the OpenSSL-specific vulnerability
4. **PBKDF2 iteration count (100k) is below OWASP 2025 recommendation** but
   irrelevant due to 128-bit Secret Key
5. **npm `pbkdf2` polyfill (CVE-2025-6545) is high risk if used** — must verify
   the web client uses native WebCrypto, not a polyfill
6. **ETH Zurich malicious-server attacks** confirm that server compromise +
   client code tampering can break vault confidentiality — aligns with the
   white paper's acknowledged weaknesses (Appendix A.2, A.3, A.5)
7. **No exploits on Exploit-DB** for 1Password
8. **All high-severity CVEs required local access** (macOS IPC, DLL injection)

### Actionable Items for Phase 2-3

1. ~~**Verify WebCrypto vs polyfill**~~ — **RESOLVED (Step 3.9)**: Native
   `crypto.subtle.deriveBits()` confirmed. CVE-2025-6545 does NOT apply.
2. **Test SRP timing** — PARASITE-class timing attack against the production
   server, even though the Go library is likely safe (Step 3.2). **Requires
   valid account credentials to get SRP parameters.**
3. **Check AES-GCM nonce generation** — verify per-item unique nonces when
   we reach vault encryption analysis (Step 3.7). **Requires authenticated
   session.**
4. **Investigate CVE-2022-32550 residual** — the SRP connection validation
   deviation was patched, but understand exactly what deviated to look for
   similar issues in the current implementation


## Phase 2: Protocol Analysis

### Step 2.1-2.2 — Auth Flow Discovery & SRP Parameter Extraction

**Date:** 2026-04-24

### Auth Endpoint Discovery

The web client JS bundle (`webapi-d3ad37f206b68333b768.min.js`) reveals a
**three-step SRP auth flow** not previously documented:

1. **`POST /api/v3/auth/start`** — SRP init (v3, not v1/v2!)
   - Request: `{email, skFormat, skid, deviceUuid, userUuid}`
   - Response: `{status, sessionID, accountKeyFormat, accountKeyUuid, userAuth: {method, alg, iterations, salt}}`
   - `encrypted: false` — no MAC needed
2. **`POST /api/v2/auth`** — SRP key exchange
   - Request: `{userA: <client_ephemeral_hex>}`
   - Response: `{userB: <server_ephemeral_hex>}`
   - `encrypted: false`
3. **`POST /api/v2/auth/confirm-key`** — SRP verification
   - Request: `{clientVerifyHash: <M1>}`
   - Response: `{serverVerifyHash: <M2>}`
   - `encrypted: false`

After successful auth, a `complete` call registers the device and receives
server config. All subsequent requests are encrypted with the session key.

### Critical Finding: `X-AgileBits-Client` Header Required

All API endpoints return `403` (text/plain) without the correct
`X-AgileBits-Client` header. The correct value is:

```
X-AgileBits-Client: 1Password for Web/2248
```

Format: `{clientName}/{clientVersion}` where:
- `clientName` = `"1Password for Web"` (set in `setDevice()`)
- `clientVersion` = build version from `data-version` HTML attribute (currently `2248`)

Without this header, all v2/v3 endpoints return `403` with empty body and
`text/plain` content-type. With the header, endpoints return proper JSON
responses with CORS, HSTS, CSP, and `x-request-id` headers.

**This was the key to unlocking API access.** Earlier reconnaissance (Step 1.4)
was testing v1 endpoints which don't require this header, but v2/v3 do.

### `skFormat` Validation

The `skFormat` field must be `"A3"` (string). Sending `"3"` (numeric format)
returns `400 {"reason":"invalid_sk_prefix_format"}` — the only descriptive
error message the server returns. All other invalid payloads return `400 {}`.

This confirms:
- The server validates Secret Key format before processing
- `"A3"` is the expected format prefix
- Only the format check produces a descriptive error; all subsequent
  validation failures return empty `{}`

### No User Enumeration

Tested 7 different email addresses plus empty string against
`/api/v3/auth/start` with valid headers and `skFormat: "A3"`:

| Email | Status | Body | Time |
|-------|--------|------|------|
| test@example.com | 400 | {} | 193ms |
| admin@1password.com | 400 | {} | 150ms |
| ctf@1password.com | 400 | {} | 170ms |
| bugbounty@agilebits.com | 400 | {} | 155ms |
| jeff@1password.com | 400 | {} | 171ms |
| security@1password.com | 400 | {} | 166ms |
| (empty) | 400 | {} | — |

All return identical `400 {}`. Timing variance is within network jitter
(~40ms range). **No user enumeration via this endpoint.**

### Auth Flow Requirements

The `_startAuth` JS function reveals:
- **Secret Key is mandatory** — throws `"Missing Secret Key"` before any
  network request
- `skid` = UUID extracted from the Secret Key itself (first segment)
- `userUuid` = the account's user UUID (stored locally from prior signin)
- `deviceUuid` = browser-generated UUID (persisted in localStorage)
- If startAuth returns `status: "device-not-registered"`, the client calls
  `registerDevice()` (which requires the session) and retries

**Implication:** Without a valid `(email, skid, userUuid)` tuple, we cannot
get SRP parameters (salt, iterations, B) from the server. The server does
not return these for unknown accounts — it returns `400 {}` with no
distinguishing information.

### Device Registration Flow

From the JS bundle:
```
if ("device-not-registered" === status) {
    setSessionUuid(sessionID);
    await registerDevice(session, device);
    // retry startAuth
}
```

Device registration happens AFTER receiving a sessionID from startAuth,
which requires valid credentials. **Cannot register a device without first
authenticating.**

### `auth/methods` Endpoint

`POST /api/v2/auth/methods` with body `{email, userUuid}`:
- Returns `{authMethods: [{type: "PASSWORD+SK"}], signInAddress: "..."}`
- Previously worked without `X-AgileBits-Client` header (Step 1.4)
- Confirms pure 2SKD auth, no SSO, no passkey-only path

### Recovery Flow Status

All recovery endpoints (`/api/v2/recovery-keys/*`) return `403` even with
the correct `X-AgileBits-Client` header. These endpoints likely require
an authenticated session or a different client identifier. **Cannot probe
recovery flow without credentials.**

### Assessment

The auth flow is tighter than initially estimated:

1. **v3 auth endpoint** was not known until JS analysis — v1/v2 probing in
   Step 1.4 was hitting wrong endpoints
2. **`X-AgileBits-Client` header** acts as a soft gatekeeper — without it,
   all modern endpoints are inaccessible
3. **SRP parameters are not exposed** without valid account identifiers —
   cannot extract salt, iterations, or server ephemeral B for unknown accounts
4. **The three-step auth flow** means SRP init and key exchange are separate
   requests, which may create opportunities for state manipulation
5. **Recovery endpoints require more than just the client header** — likely
   need an authenticated session

### What We Need to Proceed

Phase 2 (Protocol Analysis) and Phase 3 (Attack Execution) both require
SRP parameters that we can only get with valid account identifiers. Options:

1. **Obtain a test account** — if the CTF rules allow creating or using a
   provided account on `bugbounty-ctf.1password.com`
2. **Find the CTF account email/UUID** — the challenge description says
   "bad poetry" is stored in a "dedicated 1Password Bug Bounty CTF account"
   but doesn't provide login details
3. **Focus on client-side attacks** — JS bundle analysis, WASM bypass, CSP
   weaknesses (Step 3.9) don't require authentication
4. **Brute force account identifiers** — infeasible given the `(email, skid,
   userUuid)` tuple requirement with identical 400 responses


## Phase 3: Client-Side Attack Analysis

### Step 3.9 — Client-Side Attack Surface

**Date:** 2026-04-23

### PBKDF2: Native WebCrypto Confirmed (CVE-2025-6545 Does NOT Apply)

The web client uses **native `crypto.subtle`** for all PBKDF2 operations:

```javascript
// From webapi bundle — actual key derivation
const c = await o.subtle.importKey("raw", i, {name: "PBKDF2"}, false, ["deriveBits"]);
const d = await o.subtle.deriveBits({name: "PBKDF2", salt: r, iterations: s, hash: {name: e}}, c, 8*n);
```

- All PBKDF2 calls go through `crypto.subtle.importKey` + `crypto.subtle.deriveBits`
- The npm `pbkdf2` polyfill (CVE-2025-6545 / CVE-2025-6547, zero-buffer on
  non-normalized algorithm names) is **NOT used**
- HKDF uses `crypto.subtle` as well
- **CVE-2025-6545 is definitively ruled out**

**Iteration counts discovered:**
- `DEFAULT_ITERATIONS = S = 65e4` = **650,000** (higher than OWASP 2025 recommendation of 600k!)
- `ITERATIONS_100_000 = 1e5` = 100,000 (used for token-based PBKDF2)
- The Bitwarden import path also uses `crypto.subtle.deriveBits`

### Lodash 4.17.21 — Unfixed CVEs, But NOT Exploitable

**Two unfixed CVEs exist in lodash 4.17.21:**

| CVE | CVSS | Description | Fixed in |
|-----|------|-------------|----------|
| CVE-2025-13465 | 6.9 MEDIUM | Prototype pollution via `_.unset`/`_.omit` — can delete Object.prototype properties | 4.17.23 |
| CVE-2026-2950 | 6.5 MEDIUM | Bypass of CVE-2025-13465 fix via array-wrapped paths | 4.18.0 |

These enable **destructive prototype pollution** — deleting properties from
built-in prototypes via user-controlled paths to `_.unset()` or `_.omit()`.

**However:** Grep of all 4 application bundles (app, webapi, vendor-1password,
vendor-lodash) found **zero calls to `_.unset` or `_.omit`**. The vulnerable
functions are shipped in the lodash bundle but never invoked by the application.

**Assessment:** Not exploitable in the current application. The lodash bundle
is the standard full build (170KB) but the app only uses safe lodash functions
(`.merge`, `.set`, `.get`, `.pick`, etc.).

### WASM Hash Whitelist — Main Thread Only

The WASM security model (documented in Step 1.3) has a significant
architectural limitation:

**The hash whitelist monkey-patch only protects the main thread.**

Each of the 5 WASM modules in `vendor-1password` is loaded through patched
`WebAssembly.instantiateStreaming` / `WebAssembly.instantiate` calls. These
patched functions check the SHA-256 hash against the 6-hash whitelist before
allowing compilation.

**But Web Workers and Service Workers get a fresh `WorkerGlobalScope` with
the native, unpatched `WebAssembly` API.** The hash check does not exist in
Worker contexts.

**Worker infrastructure discovered:**
- Worker scripts served from document origin: `https://bugbounty-ctf.1password.com/workers/` (HTTP 200)
- Workers loaded via: `new Worker(new URL(\`https://${host}/${workersDir}${name}\`).href)`
- `worker-src 'self'` in CSP restricts Workers to same origin only
- Firebase messaging service worker at `/firebase-messaging-sw.js` (HTTP 200)
  - Imports Firebase SDK from `app.1password.com/libjs/`
  - Handles push notification events

**Attack chain (theoretical):**
1. Find a way to inject or replace a same-origin Worker script
2. Inside the Worker, call native `WebAssembly.instantiate()` with arbitrary
   WASM bytecode — no hash check runs
3. The malicious WASM module has full access to the Worker's memory and
   can communicate back via `postMessage`

**Mitigations that block this chain:**
- `worker-src 'self'` — no blob: or data: URLs for Workers
- SRI on all script tags (but Workers bypass SRI since they're not `<script>` tags)
- Worker scripts are static, served from the CDN proxy
- No file upload endpoint discovered that could create a Worker script

**Verdict:** Theoretical bypass exists but requires a prerequisite vulnerability
(same-origin script injection or service worker hijacking) that has not been found.

### postMessage Handlers — Origin Validation Analysis

**5+ `message` event listeners identified across bundles.**

#### Validated handlers (safe):
- **StripeFrame**: `origin === u` where `u` is the Stripe payment URL ✅
- **DuoFrame**: `t.origin === i` where `i = "https://duo.1passwordservices.com"` ✅

#### Weakly validated handlers:

**Idle timer reset handler:**
```javascript
// Accepts messages from any *.1password.com subdomain
const n = e.origin.endsWith("." + M.S9.config.server);
const t = !!e.source?.opener && e.source.opener === window;
const o = "reset_idle_timer" === e.data.type;
n && t && o && this.resetIdleTimer();
```
Three conditions required: origin endsWith `.1password.com`, source is a
window opened by this window, and message type is `reset_idle_timer`. The
opener check (`e.source.opener === window`) prevents arbitrary cross-origin
abuse — only a popup opened by this specific window can send the message.

**L handler (extension communication):**
```javascript
D = (e, t) => {
    const n = new URL(t);
    const o = e.config.server;
    return "" !== o && n.host.slice(n.host.indexOf(".") + 1) === o;
};
```
Origin validation extracts everything after the first dot in the host. For
`evil.1password.com`, this yields `1password.com` which matches `config.server`.
**Any `*.1password.com` subdomain passes this check.**

If a subdomain takeover exists on any `*.1password.com` domain (e.g., a
dangling CNAME to an unclaimed cloud resource), an attacker could:
1. Take over the subdomain
2. Open the 1Password web client in an iframe (allowed by `frame-ancestors`)
3. Send postMessage with a controlled payload
4. The `L` handler accepts it because the origin passes `D()` validation

The `L` handler dispatches to various flows (DelegatedSession, SingleSignOn,
etc.) — if any of these flows can be triggered externally, this could be
significant.

#### Wildcard postMessage (info leak):
```javascript
window.opener && window.opener.postMessage({READY: true}, "*");
```
When the signin page loads, it sends `{READY: true}` to its opener window
with `"*"` as the target origin. **Any page that opens the signin page via
`window.open()` receives this message.** This is a minor info leak: an
attacker can detect when the signin page has finished loading. Not directly
exploitable for data exfiltration, but could be used as a timing oracle in
a more complex attack chain.

### Service Worker Analysis

**Firebase messaging service worker** (`/firebase-messaging-sw.js`):
- Registered via `navigator.serviceWorker.register("firebase-messaging-sw.js")`
- Imports Firebase SDK via `importScripts` from `app.1password.com`
- Handles push notifications with custom `NotificationEvent` class
- Has full `WebAssembly` API in its scope (unpatched)
- Service Worker scope: root (`/`) — intercepts all fetch events

**No other service workers found.** The app does not register a custom
service worker for offline caching or request interception.

### Summary Assessment

| Finding | Severity | Exploitable? |
|---------|----------|-------------|
| PBKDF2 uses native WebCrypto (CVE-2025-6545 N/A) | — | No |
| Lodash 4.17.21 has unfixed CVEs (CVE-2025-13465, CVE-2026-2950) | Medium | **No** — `_.unset`/`_.omit` not called |
| WASM hash check is main-thread only | Medium | **Theoretical** — requires same-origin script injection |
| postMessage `D()` accepts any `*.1password.com` subdomain | Low | **Conditional** — requires subdomain takeover |
| `window.opener.postMessage({READY:true}, "*")` | Info | Minor timing oracle |
| Firebase service worker has unpatched WASM API | Low | **Theoretical** — requires SW hijacking |
| PBKDF2 default iterations = 650,000 | — | Exceeds OWASP 2025 recommendation |

**The client-side attack surface is well-defended.** The combination of SRI,
strict CSP, WASM hash whitelist, and proper origin validation on critical
postMessage handlers leaves no directly exploitable path without first
obtaining a prerequisite vulnerability (subdomain takeover or same-origin
script injection).

### Step 3.10 — Pre-Auth Endpoint Probing

**Date:** 2026-04-23

#### Recovery Flow Architecture (from JS Bundle)

The recovery flow uses its own SRP handshake, separate from the main auth flow.
All three initial steps are `encrypted: false`:

```
1. POST /api/v2/recovery-keys/session/new
   Body: {recoveryKeyUuid: string}
   Response: {sessionUuid: string, cryptoVersion: string}

2. POST /api/v2/recovery-keys/session/auth/cv1/start
   Body: {bigA: string}  (SRP client ephemeral A)
   Response: {bigB: string}  (SRP server ephemeral B)

3. POST /api/v2/recovery-keys/session/auth/cv1/confirm
   Body: {clientHash: string}  (SRP M1)
   Response: {serverHash: string}  (SRP M2)
```

Steps 4+ (email verification, material retrieval, completion) are `encrypted: true`
— they require the session key from the SRP handshake.

**Recovery key structure:**
```
{uuid, label, enc, encryptedBy, cryptoVersion, verifierParam}
```

The `verifierParam` field is the SRP verifier for this recovery key. Each
recovery key acts as a separate SRP credential, independent of the account
password + Secret Key.

**Testing:** Both `00000000-0000-0000-0000-000000000000` and random UUIDs
return identical `400 {}`. No timing difference (both ~125-155ms, within
network jitter). Recovery key UUID enumeration is not feasible.

#### Pre-Auth Endpoint Scan Results

| Endpoint | Method | Status | Response | Notes |
|----------|--------|--------|----------|-------|
| `/api/v2/preauth-perftrace` | PUT | **200** | `{"success":1}` | Accepts ANY body including empty — write-only telemetry sink |
| `/api/v2/preauth-perftrace` | POST | 405 | — | Method not allowed |
| `/api/v2/preauth-perftrace` | GET | 405 | — | Method not allowed |
| `/api/v2/perftrace` | POST | 405 | — | |
| `/api/v1/monitoring/status` | GET | 401 | `{}` | |
| `/api/v2/signinattempts` | GET | 405 | — | |
| `/api/v1/signinattempts` | POST | 405 | — | |
| `/api/v1/confidential-computing/session` | POST | 422 | `{"reason":"Failed to parse..."}` | Descriptive error with column number |
| `/api/v2/session-restore/save-key` | POST | 401 | `{}` | |
| `/api/v2/session-restore/destroy-key` | POST | 405 | — | |
| `/api/v1/signup` | POST | 400 | `{}` | Signup disabled |
| `/api/v2/signup` | POST | 400 | `{}` | Signup disabled |
| `https://flow.1passwordservices.com/` | GET | 403 | `{"message":"Missing Authentication Token"}` | AWS API Gateway |

#### Confidential Computing Endpoint

`POST /api/v1/confidential-computing/session` returns a **descriptive
Rust serde error** with specific column numbers:

```
{"reason":"Failed to parse the request body as JSON at line 1 column 22"}
```

This is an `encrypted: false` endpoint (from JS analysis). The error format
confirms a **Rust backend** (serde_json error format). The column numbers
shift based on the specific fields sent — the parser successfully reads
the JSON but rejects the structure because required fields are missing or
types are wrong.

This is the only endpoint that returns a descriptive reason in the error
body (besides the `skFormat` validation error discovered in Step 2.1).

#### Secret Key Retrieval Fallback Script

The `sk-2c17b526b1a01ed2f995.min.js` script (54KB) is loaded only when the
main app fails to render (`displayFallback()`). It contains:
- Custom big number library (not WASM-based)
- Standalone SRP implementation (no WebCrypto dependency)
- Used to retrieve the Secret Key in degraded browser environments

This fallback crypto code does NOT go through the WASM hash whitelist or
the main app's crypto pipeline. If an attacker could force the fallback
condition (e.g., by causing the main app scripts to fail), the fallback
SRP code would run without WASM protections. However, the SRP security
properties should be equivalent — the fallback just uses a different
implementation (JS BigInt vs. WASM).

#### Assessment

No exploitable pre-auth endpoints found. Key observations:

1. **`preauth-perftrace`** is a write-only sink — cannot read back data
2. **Recovery flow requires a valid `recoveryKeyUuid`** — UUID space is
   too large to enumerate, responses are identical for all invalid UUIDs
3. **Confidential computing** leaks implementation detail (Rust backend)
   but requires specific structured input we don't have the schema for
4. **Signup is disabled** on the CTF instance — cannot create accounts
5. **All authenticated endpoints return uniform `401 {}`** — no info leaks

### Overall Phase 3 Pre-Auth Assessment

**All pre-auth attack vectors have been exhausted without finding an
exploitable vulnerability.**

| Vector | Status | Verdict |
|--------|--------|---------|
| Client-side JS exploitation | Tested | No XSS vector within CSP constraints |
| WASM module substitution | Tested | Hash whitelist blocks on main thread; Worker bypass theoretical only |
| Lodash prototype pollution | Tested | Vulnerable functions (`_.unset`/`_.omit`) never called |
| postMessage origin bypass | Tested | Requires subdomain takeover (not found) |
| PBKDF2 polyfill weakness | Tested | Native WebCrypto used; CVE-2025-6545 N/A |
| Recovery flow enumeration | Tested | Uniform 400 responses, no timing leak |
| Pre-auth endpoint info leak | Tested | Only `preauth-perftrace` (write-only) returns 200 |
| Signup / account creation | Tested | Disabled on CTF instance |
| User enumeration | Tested | Uniform responses across all tested emails |

**To proceed further, the engagement needs either:**
1. Valid account credentials (email + Secret Key + password)
2. A previously undiscovered pre-auth vulnerability
3. A subdomain takeover on `*.1password.com` (would enable postMessage attack)
4. Access to the server-side infrastructure (out of scope for this CTF)


### Step 3.11 — Crypto Architecture Analysis (from JS Bundle)

**Date:** 2026-04-23

Full key hierarchy reconstructed from the `webapi` and `vendor-1password`
bundles without requiring authentication.

#### Key Derivation Chain

```
Password ──► PBKDF2-HMAC-SHA256 (650k iterations) ──► kdfBytes (32 bytes)
                                                         │
Secret Key ──► HKDF-SHA256(key=SK, salt=format, info=id) ──► personalBytes
                                                         │
                                                    XOR ─┤
                                                         │
                                                    combinedKey (32 bytes)
                                                    ┌────┴────┐
                                                    ▼         ▼
                                              AUK (mp)   SRP-x
                                         (vault unlock)  (auth)
```

**2SKD `combineWithBytes` implementation:**
```javascript
combineWithBytes = async (derivedBytes) => {
    rawKey = this.rawKeyMaterial;        // Secret Key raw bytes
    format = this.format.toString();      // "A3"
    id = this.id;                         // Key UUID
    personal = await HKDF(SHA256, rawKey, format, id, derivedBytes.length);
    return XOR(derivedBytes, personal);   // byte-wise XOR
}
```

The Secret Key contributes entropy through HKDF (not raw XOR), using the
format string as salt and key UUID as info. Both factors (PBKDF2 output +
HKDF output) must be known to reconstruct `combinedKey`.

#### SRP-x Computation

```javascript
x = SHA1(salt || SHA1(email || ":" || hex(combinedKey)))
```

Uses **SHA-1** (160-bit output) per RFC 5054 standard SRP-6a. SHA-1's
collision resistance weakness is irrelevant here — SRP-x only requires
preimage resistance, which SHA-1 still provides. The 160-bit x is fed
into `v = g^x mod N` (the SRP verifier).

#### AES-256-GCM Vault Encryption

```javascript
encrypt = async (plaintext, aad?) => {
    iv = getRandomBytes(12);         // 96-bit random nonce via CSPRNG
    ciphertext = AES-GCM-256(key, iv, plaintext, tagLength=128, aad);
    return {data: ciphertext, iv: iv};
}
```

- **Algorithm:** AES-256-GCM via native `crypto.subtle`
- **IV:** 12 bytes (96 bits) generated by `crypto.getRandomValues()`
- **Tag length:** 128 bits (full GCM tag, no truncation)
- **AAD:** Optional additional authenticated data
- **Key generation:** 32 random bytes via `crypto.getRandomValues()`

Random nonces have a birthday bound collision risk at ~2^48 encryptions
per key. For vault items, this is not a practical concern.

#### Key Hierarchy

```
Account Password + Secret Key
         │
         ▼
   combinedKey (2SKD output)
    ┌────┼────┐
    ▼    │    ▼
  AUK    │   SRP-x ──► SRP verifier v = g^x mod N
  (mp)   │              (stored server-side)
    │    │
    ▼    ▼
  Primary Keyset
    │
    ├──► Vault Keys (per-vault, wrapped by keyset)
    │      │
    │      ├──► Item Keys (per-item AES-256-GCM)
    │      │
    │      └──► Attachment Keys
    │
    ├──► Credential Bundle (SRP-x + AUK, for device auth)
    │
    └──► Recovery Key (optional, separate SRP verifier)
```

#### Observations

1. **No shortcuts exist in the key hierarchy.** Recovering vault items
   requires either the AUK (which requires password + Secret Key) or
   a server-side vulnerability that exposes encrypted blobs AND the
   wrapping keys.

2. **SRP verifier is one-way.** Even with the server's SRP verifier `v`,
   recovering `x` requires solving the discrete logarithm problem mod N
   (a 2048-bit safe prime). This is computationally infeasible.

3. **All cryptographic operations use native `crypto.subtle`** — no
   JavaScript crypto polyfills, no custom RNG, no weak primitives.

4. **The WASM modules in `vendor-1password`** implement the performance-
   critical crypto operations (likely HKDF, key wrapping, perhaps the
   SRP big number operations for the full app path). The fallback `sk`
   script uses SJCL (Stanford JavaScript Crypto Library) for the same
   operations in JS.

5. **Recovery keys have independent SRP verifiers** — compromising a
   recovery key's SRP handshake would only authenticate the recovery
   session, not reveal the vault encryption key. The vault keys are
   still encrypted by the primary keyset, which requires the original
   password + Secret Key to derive.

#### Verdict

The crypto architecture has no theoretical weaknesses exploitable without:
- **Quantum computing** (breaks DLP for SRP, AES-128 effective for Grover's)
- **Implementation bug** (timing leak in server-side modular exponentiation)
- **Server compromise** (access to encrypted blobs + verifier database)
- **Client compromise** (XSS/code injection to capture keys in memory)

All four of these are either out of scope, mitigated by the client-side
protections documented in Step 3.9, or require access we don't have.
