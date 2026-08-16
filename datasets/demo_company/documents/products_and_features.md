<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
NovaCloud product catalog and feature inventory. All product
names, code names, and feature flags are fictional.
-->

# NovaCloud — Products and Features

NovaCloud offers three core products and several add-on modules.
The three core products are **NovaCompute**, **NovaDB**, and
**NovaStream**. The add-on modules are **NovaObserve**,
**NovaShield**, and **NovaEdge**. This document describes
each one at a high level. Detailed pricing is in the Pricing
Plans document.

## NovaCompute

NovaCompute is the company's flagship container orchestration
service. It abstracts away Kubernetes cluster management so
that customers can deploy a Docker image with a single CLI
command and receive a public HTTPS endpoint in under ninety
seconds. NovaCompute supports rolling deployments, blue-green
deployments, and canary deployments.

A typical NovaCompute deployment consists of three things: a
container image, a port number, and an environment configuration.
The environment configuration is a YAML file that specifies
environment variables, secrets, and resource limits. Secrets
are encrypted at rest using per-customer AES-256 keys and
never appear in logs.

NovaCompute has three deployment tiers:

- **Edge tier**: workloads run in shared infrastructure
  optimized for low cold-start latency.
- **Standard tier**: workloads run in dedicated nodes with
  predictable performance.
- **Isolated tier**: workloads run on single-tenant nodes,
  suitable for regulated workloads.

Each tier supports a different set of resource ceilings. See
the Usage Limits document for current numbers.

## NovaDB

NovaDB is a managed PostgreSQL-compatible database service.
It supports point-in-time recovery, read scaling up to six
replicas, and automatic failover under thirty seconds. It is
based on a hardened fork of PostgreSQL 16, with patches
contributed back to the upstream community when they pass
review.

NovaDB clusters can be created in the `eu-west-1`, `us-east-1`,
`us-west-2`, and `ap-southeast-1` regions. Cross-region
replication is available on the Enterprise plan only.

## NovaStream

NovaStream is a managed event streaming service compatible
with the Kafka wire protocol. It supports topics with up to
ninety days of retention and consumer groups with up to five
hundred active consumers. NovaStream is often used by
customers for change-data-capture pipelines out of NovaDB.

NovaStream is sometimes confused with "NovaStream Analytics,"
which is a separate product that runs scheduled SQL queries
over NovaStream topics. The two share infrastructure but have
separate billing. Customers should consult the Pricing Plans
document before enabling NovaStream Analytics on production
topics.

## NovaObserve

NovaObserve is an observability add-on that ingests metrics,
logs, and traces from NovaCompute, NovaDB, and NovaStream. It
provides a query language similar to PromQL and a dashboard
editor. NovaObserve can also ingest data from external sources
via an OpenTelemetry-compatible endpoint.

## NovaShield

NovaShield is a Web Application Firewall and DDoS protection
service. It runs in front of any public NovaCompute endpoint
and applies a configurable rule set. NovaShield blocks
approximately 14 million malicious requests per day across
the entire NovaCloud fleet, based on the most recent quarterly
threat report.

## NovaEdge

NovaEdge is a CDN service that caches static assets at 187
edge locations worldwide. NovaEdge supports custom SSL
certificates, custom cache keys, and origin shield nodes. It
is integrated with NovaCompute so that any container serving
traffic can be fronted by NovaEdge with a one-line manifest.

## Feature flags

The following feature flags are currently in private beta and
will roll out to general availability over the next two quarters:

- **GPU pooling** for NovaCompute Standard and Isolated tiers.
- **Cross-region live migration** for NovaDB.
- **Schema diff** tool for NovaDB migrations.
- **Inline transforms** for NovaStream.

Customers interested in early access should contact their
account executive. Early-access features are subject to
different SLAs than generally-available features.

## Distractor

Customers frequently ask whether NovaCompute supports Windows
serverless containers. It does not. Windows container support
is on the roadmap but has been deferred three times; the most
recent deferral was announced at the November 2024 customer
advisory board meeting and is not expected before Q3 2026.
Until then, customers running Windows workloads should use
NovaCompute Standard tier with dedicated Windows nodes, which
are billed at a different rate. See the Pricing Plans document
for the dedicated-Windows rate.