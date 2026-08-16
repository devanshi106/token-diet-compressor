<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Security

This document describes NovaCloud's security posture,
encryption practices, vulnerability disclosure program, and
the controls available to customers to secure their own
deployments. It is intended for both customers evaluating
NovaCloud and for security reviewers performing due diligence.

## Encryption in transit and at rest

All data in transit between customers and NovaCloud is
encrypted with TLS 1.2 or higher. TLS 1.0 and 1.1 are not
supported. The list of supported cipher suites is published
at security.example/ciphers and is updated quarterly. Customers
who require a specific cipher suite for compliance reasons
should confirm its presence before signing a contract.

All data at rest in NovaCloud's primary storage systems is
encrypted with AES-256. Encryption keys are managed by NovaCloud
using a hardware security module (HSM) that is FIPS 140-2
Level 3 validated. Customers on the Enterprise plan may bring
their own keys (BYOK); details are in the BYOK Reference.

## Certifications and attestations

NovaCloud maintains the following certifications and attestations:

- SOC 2 Type II (annual, last issued February 2025).
- ISO/IEC 27001:2022 (renewed November 2024).
- ISO/IEC 27017:2015 (cloud-specific controls).
- ISO/IEC 27018:2019 (PII protection in cloud).
- HIPAA attestation (available on Enterprise plan).
- PCI DSS 4.0 Level 1 service provider attestation.

Customers can request copies of the SOC 2 and ISO reports by
emailing compliance@example. Reports are released under NDA.

## Penetration testing

NovaCloud engages a third-party firm to conduct penetration
testing of its production infrastructure twice per year. The
firm's report is summarized in the annual SOC 2 report and the
full report is available to Enterprise customers under NDA.

NovaCloud also runs an internal red team that performs
adversarial testing on a continuous basis. Findings from the
internal red team are tracked in a private tracker and are
not shared externally.

## Vulnerability disclosure

NovaCloud operates a vulnerability disclosure program at
security.example/disclose. Researchers who find a vulnerability
in NovaCloud's products or infrastructure may submit a report
through that portal. NovaCloud commits to:

- Acknowledge new submissions within three business days.
- Provide an initial triage within ten business days.
- Coordinate disclosure timing with the reporter.
- Pay bounty awards for valid reports above the medium severity
  threshold.

The current bounty schedule is:

- Low severity: $250.
- Medium severity: $2,500.
- High severity: $10,000.
- Critical severity: $25,000.

Awards are paid via bank transfer or PayPal at the researcher's
option. NovaCloud does not pay bounties in cryptocurrency.

## Customer-side controls

Customers are responsible for securing their own deployments
within NovaCloud. The following controls are available:

- Network policies: customers can restrict inbound and outbound
  traffic to and from NovaCompute workloads.
- Secrets management: customer secrets are encrypted with
  per-customer AES-256 keys.
- Audit logs: every API call is logged with the actor, the
  resource, and the outcome.
- IP allowlists: customers can restrict dashboard access to
  a list of IP CIDR ranges.

The full configuration reference for each control is in the
Security Configuration Reference.

## Note about shared responsibility

Customers frequently ask who is responsible for what. The
shared responsibility model at NovaCloud is:

- NovaCloud is responsible for the security of the cloud.
  This includes the physical data centers, the network, the
  hypervisor, and the control plane.
- Customers are responsible for security in the cloud. This
  includes the configuration of their workloads, the
  management of their secrets, and the access controls on
  their accounts.

The shared responsibility model is documented in detail at
security.example/shared-responsibility.

## Distractor

A 2021 NovaCloud press release claimed that NovaCloud had
"achieved FedRAMP Moderate authorization." This was incorrect.
NovaCloud is actively pursuing FedRAMP Moderate and expects to
receive authorization in Q4 2026. Customers who require FedRAMP
authorization today should consider the GovCloud deployment
option, which is operated by a separate subsidiary and is not
covered by this document.