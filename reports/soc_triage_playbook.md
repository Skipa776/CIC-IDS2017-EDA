# SOC Triage Playbook — CICIDS2017 Layered IDS

**Scope:** alert triage for the two-layer LightGBM pipeline served by `api/` (layer 1 flags
attacks, layer 2 names the attack type, responses include the MITRE ATT&CK mapping).
**Provenance:** every number traces to `models/model_metadata.json` (v2.0.0, trained 2026-06-12),
`reports/attack_clusters.json` (KMeans k=14 behavioral clusters), and
`api/services/mitre_mapping.py`. Coverage: 15 traffic labels → **8 distinct ATT&CK techniques
across 6 tactics**; clustering surfaced 14 behavior clusters dominated by 6 of those techniques.

**Confidence levels in API responses:** high ≥ 0.90, medium ≥ 0.70, low < 0.70
(probability of the predicted class).

---

## 1. What to trust — and what not to

**Trust (random-split test, natural prevalence):** for attack types that were in the training
data, layer 1 catches ≥ 93% of flows in every family with more than 9 test flows (Infiltration,
n=9, is 0.67; SQL injection, n=5, is 0.60) at a 0.4% false-positive rate on benign traffic.
This does not hold for attack types the model has not seen (see the next table). Layer 2 names
DoS/DDoS/PortScan/brute-force types with F1 ≥ 0.99.

**Do NOT trust blindly:**

| Blind spot | Evidence | Operational consequence |
|---|---|---|
| **Novel attack behavior** | Cross-day holdout PR-AUC drops to **0.824** (Friday attacks unseen) and **0.466** (DoS-family/web/Heartbleed unseen). At the default 0.5 threshold, recall is **0.011** (Friday) and **0.041** (Wed+Thu). | A "no alerts" day does not mean a clean network for attack types the model never trained on. Keep signature/WAF/EDR controls primary for novelty. |
| **Web-attack type identification** | Layer 2 F1: XSS 0.51, Brute Force 0.58 (recall 0.45), SQLi 0.67 | Layer 1 will flag web attacks (~97% recall), but the *named type* is unreliable — confirm via WAF/application logs before acting on the label. |
| **Ultra-rare classes** | Heartbleed n=1, Infiltration n=9, SQLi n=5 in the test set | Metrics are anecdotal; treat any such alert as unvalidated and escalate manually. |
| **Infiltration recall 0.67** | Layer 1 per-attack table | Lowest binary recall of any family; pair with egress monitoring/DLP (see §2.8). |

---

## 2. Per-technique triage

Severity guidance assumes a single flagged flow; sustained volumes raise severity one level.
Mitigations are abbreviated from `api/services/mitre_mapping.py` (the API returns the full list).

### 2.1 T1498.001 — Network DoS: Direct Network Flood (Impact)
**Labels:** DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest · **Layer-1 recall:** ≥ 0.997
**Behavioral signatures (clusters 1, 4–6, 8, 10–12):** two distinct modes —
*volumetric* (Hulk/GoldenEye: backward packet-length std z ≈ +8–11 vs benign) and
*slow-rate* (slowloris/Slowhttptest: inter-arrival and active/idle times z ≈ +9–18).
**Triage:** confirm service degradation (latency/5xx); identify top source IPs; check whether
slow-rate (connection exhaustion) or volumetric (bandwidth).
**Respond:** rate limiting + per-IP connection caps; aggressive HTTP header/body timeouts
(5–10 s) for slow-rate; reverse proxy buffering (nginx/HAProxy); M1037 Filter Network Traffic.

### 2.2 T1498 — Network Denial of Service / DDoS (Impact)
**Label:** DDoS · **Layer-1 recall:** 1.000
**Signature:** volumetric flood pattern across many sources (clusters overlap with Hulk-style
behavior — distributed origin is the differentiator, which flow features alone do not show).
**Triage:** source-IP cardinality is the key pivot; single-source = T1498.001 instead.
**Respond:** upstream scrubbing (M1037), anycast/CDN absorption (M1035), BGP flowspec; engage
the DDoS runbook rather than per-IP blocking.

### 2.3 T1046 — Network Service Scanning (Discovery)
**Label:** PortScan · **Layer-1 recall:** 1.000
**Signatures (clusters 2, 13):** minimal payloads (average packet size *below* benign), bare
SYN/PSH probing, high backward packets/s (z ≈ +26 in the fast-scan cluster). Cluster 2 is the
low-purity catch-all (23% pure) — scan-like flows blend with other recon traffic.
**Triage:** scope = how many destination ports/hosts from the source within the window;
internal sources are a potential lateral-movement precursor — raise severity.
**Respond:** block/quarantine source; M1030 Network Segmentation; M1031 IPS scan-detection
rules; review exposed services it enumerated.

### 2.4 T1110.001 — Brute Force: Password Guessing (Credential Access)
**Labels:** FTP-Patator (recall 0.997), SSH-Patator (0.997), Web Attack – Brute Force (0.970;
but layer-2 type naming only 0.45 reliable)
**Signature (cluster 0):** repeated short authentication bursts — SYN/PSH flag counts z ≈ +4.3.
**Triage:** pull auth logs for the target service; distinguish spray (many users, few passwords)
vs targeted; check for any *successful* login from the source after the burst — that converts
this to an active-compromise incident.
**Respond:** M1036 account lockout, M1032 MFA, fail2ban; for SSH disable password auth (keys
only); for web add CAPTCHA + endpoint rate limits.

### 2.5 T1190 — Exploit Public-Facing Application (Initial Access)
**Labels:** Heartbleed (n=1 test — anecdotal), Web Attack – Sql Injection (layer-1 recall 0.60, n=5)
**Signature (cluster 3, Heartbleed):** wildly oversized backward payloads (max backward packet
length z ≈ +18) — the memory-leak response pattern.
**Triage:** treat ANY alert here as high severity; verify OpenSSL/CVE-2014-0160 exposure or
inspect the offending request in app logs for SQLi payloads.
**Respond:** M1051 patch immediately; rotate TLS certs/keys after Heartbleed; parameterized
queries + WAF SQLi rules (M1050/M1021); least-privilege DB accounts.

### 2.6 T1059.007 — Command & Scripting: JavaScript / XSS (Execution)
**Label:** Web Attack – XSS · **Layer-1 recall:** 0.977, **but layer-2 naming F1 only 0.51**
**Triage:** model labels XSS/brute-force/SQLi interchangeably too often — confirm with WAF or
application logs before classifying the incident.
**Respond:** CSP headers (M1021), output encoding, HTTPOnly/Secure cookies, WAF XSS rules.

### 2.7 T1071.001 — Application Layer Protocol: Web (C2 / Bot)
**Label:** Bot · **Layer-1 recall:** 0.934
**Signature (cluster 9, 100% pure):** anomalously large, uniform forward payloads
(forward packet-length mean z ≈ +19) — beacon/upload-like behavior.
**Triage:** check destination reputation; look for regular-interval connections (beaconing)
from the same host; an internal source means assume compromise.
**Respond:** isolate host (M1030), EDR sweep, DNS sinkhole known C2 domains, block destination
(M1031).

### 2.8 T1071 — Application Layer Protocol / Infiltration (C2, Exfiltration)
**Label:** Infiltration · **Layer-1 recall: 0.67 — weakest detection in the system** (n=9)
**Signature (cluster 7, 100% pure):** extreme outbound volume (total forward bytes z ≈ +174)
over long-duration flows — bulk exfiltration shape.
**Triage:** treat as potential data theft: identify what the source host can reach, volume
transferred, and destination; this family is the model's weakest — proactive egress analytics
are the primary control, not this alert stream.
**Respond:** DLP (M1031 monitoring + egress filtering), micro-segmentation (M1030), proxy
logging review.

---

## 3. Standing guidance

1. **Every "attack" response from the API carries the MITRE mapping** — route the `mitigations`
   list into the ticket automatically.
2. **Low-confidence (< 0.70) layer-2 labels:** trust layer 1 ("it's an attack"), not the type;
   triage from raw flow features and service logs.
3. **Threshold context:** at the default 0.5 binary threshold the FP rate is 0.4% — on this
   network profile ≈ 1 false alert per ~250 benign flows flagged volume-wise; tune per traffic volume.
4. **Retraining note:** any change to `data/processed/cicids2017_clean_v2.parquet` requires
   `python scripts/train_models.py` and re-running `pytest tests/` before deploying.
