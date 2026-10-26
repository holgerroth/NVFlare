# Future governance track: Passports and DUO

This document records integration points for a later phase. The synthetic-data hackathon implements no Passport issuer, Visa verifier, DUO policy engine, or governance demo. None is required to join the pilot.

## Responsibilities

| Component | Role |
|---|---|
| [GA4GH Passports and Visas](https://www.ga4gh.org/product/ga4gh-passports/) | Convey verifiable researcher attributes and acquired data-access permissions |
| [Data Use Ontology](https://www.ga4gh.org/product/data-use-ontology-duo/) | Describe dataset use permissions and modifiers in a common machine-readable vocabulary |
| Institutional authorization service | Decide whether the authenticated researcher and proposed use may access particular datasets |
| NVFLARE studies | Scope federation membership and operational roles |
| WES and TES | Submit and manage computation under the institution's authorization decisions |

Passports provide evidence for a decision. DUO supplies vocabulary for evaluating purpose compatibility. Neither replaces institutional policy, consent interpretation, researcher login, or enforcement. Keeping data at a site does not itself establish permission to compute on it or release results.

## Proposed integration points

1. **Submission:** Authenticate the researcher in the platform starting the NVFLARE recipe (or at the optional inbound WES entry point) and validate relevant Visas against institution-approved issuers. Associate the researcher, project, declared purpose, requested datasets, and approved analysis with the run.
2. **Delegation:** Preserve authenticated researcher attribution from the recipe submission through NVFLARE to each site WES workflow. If using the optional inbound adapter, preserve attribution when its service account submits to NVFLARE. Define an institution-approved delegation mechanism binding that context to the job and workload. Self-asserted WES tags or job metadata are not proof of identity or access rights.
3. **Site authorization:** Each participating institution evaluates the authorization evidence, dataset DUO annotations, proposed purpose, and any additional local conditions before releasing data or permitting computation. Unknown or unsupported restrictions require an explicit institutional decision.
4. **Execution and results:** Permit the approved workload through the site-controlled data or execution service. Apply the institution's output-release rules before results leave the site.
5. **Audit:** Correlate the authorization decision and policy version with each site WES run ID, NVFLARE job/study/site, and any TES task IDs. Record decision references and relevant provenance without placing bearer credentials in exported jobs or logs.

## Design requirements for that phase

- Agree on trusted issuers, accepted Visa types, signature and validity checks, and the semantics of access permissions with each institution.
- Specify credential expiry, revocation, refresh, and reevaluation during long-running analyses. Define how withdrawal stops queued/running TES tasks and affects result release.
- Define delegated credential scope and audience for each service, separately from NVFLARE participant certificates. Minimize the identity attributes shared across sites.
- Establish dataset identifiers, versioned DUO annotations, ontology-aware matching, and handling of additional terms. A matching label alone is insufficient evidence of permission.
- Map approved access to existing NVFLARE study enrollment and operational roles without treating study membership as blanket dataset authorization.
- Cover direct submissions as well as WES submissions so the entry point cannot be bypassed. A check inside user-supplied `client.py` is not a security boundary.
- Test allowed and denied purposes, untrusted/expired/revoked Visas, tampered attribution, unavailable policy services, delegated workload identity, and cancellation after access withdrawal.

The primary prototype uses site-configured WES credentials alongside existing NVFLARE identities. The optional inbound adapter uses a shared caller token and one NVFLARE service identity. Neither path yet carries a verified end-user identity between these systems. A production design must close that gap together with participating institutions before sensitive-data use.
