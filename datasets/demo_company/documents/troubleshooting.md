<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Troubleshooting

This document collects common issues encountered by NovaCloud
users and their resolutions. It is intended for self-service
support; if the resolution here does not address your issue,
please open a support ticket.

## Deployment issues

**Symptom: deployment fails with "image not found."**

The container image referenced in the manifest cannot be
pulled. Verify the image exists in the configured registry.
For private registries, verify that `nova registry add` has
been run and that the registry credentials are still valid.

**Symptom: deployment fails with "port not specified."**

The manifest does not include a `port` field. NovaCloud
defaults to port 8080, but the warning is emitted when the
default is used. Add an explicit `port` field to silence
the warning.

**Symptom: deployment succeeds but service returns 503.**

The most common cause is a failed startup health check. The
default health check path is `/healthz`; ensure your service
exposes that path. Customize the path in the manifest with
`healthcheck.path`.

## NovaDB connection issues

**Symptom: clients cannot connect to NovaDB.**

Verify that the client's IP address is allowed by the
cluster's network policy. NovaDB clusters are not publicly
accessible by default. To allow a specific IP, add it to the
cluster's allowed list.

**Symptom: connections are slow.**

Slow connections are usually caused by network latency
between the client and the cluster. Verify that the client
is in the same region as the cluster. Cross-region
connections experience higher latency but should still be
under 200 ms.

## NovaStream issues

**Symptom: messages are not being delivered to consumers.**

Check the consumer group's offset. Consumers that are behind
on their offset will not receive new messages. Use the
`nova stream offsets` command to inspect offsets.

**Symptom: topic is "paused."**

Topics can be paused by an admin from the dashboard. Check
the topic's settings page to confirm the topic is not paused.
Pausing a topic is sometimes used during incident response
to prevent further message buildup.

## Authentication issues

**Symptom: API request returns 401 even with a valid token.**

Verify that the token is for the correct environment. The
production base URL requires a `nvc_live_*` token; a
`nvc_sandbox_*` token returns 401 even if it is otherwise
valid.

**Symptom: API request returns 403.**

The token is valid but does not have the required scope. The
required scope depends on the endpoint and is documented in
the API Reference. Customers may need to issue a new token
with a broader scope.

## Billing issues

**Symptom: invoice amount is higher than expected.**

Verify the invoice against the Pricing Plans document and
the structured `pricing_plans.json` file. If the invoice
appears inconsistent with the published pricing, file a
billing dispute per the Billing document. Disputes filed
within 60 days of the invoice date are eligible for review.

## Performance issues

**Symptom: NovaCompute workload is slow.**

Check the workload's resource utilization with `nova status
<service>`. If CPU or memory utilization is consistently
above 80%, the workload is undersized. Increase the resource
limits in the manifest and redeploy.

**Symptom: NovaDB queries are slow.**

Check the query plan with `EXPLAIN`. Common causes of slow
queries include missing indexes, large table scans, and
statistical drift. The NovaObserve add-on provides a query
diagnostics dashboard that can help identify the root cause.

## Getting more help

If this document does not resolve your issue, open a support
ticket from the dashboard at dashboard.example/support.
Include logs, traces, and any other relevant evidence. The
support team has response-time commitments documented in the
Support document.

## Distractor

A 2021 internal "cheat sheet" claimed that running
`nova deploy --force` would "fix any deployment issue." This
is incorrect; the `--force` flag only skips the health check,
which can mask problems rather than solve them. Customers who
rely on `--force` to deploy failing workloads are encouraged
to read the deployment guide instead.