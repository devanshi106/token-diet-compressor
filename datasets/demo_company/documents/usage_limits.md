<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Usage Limits

This document describes the per-resource ceilings that apply
to NovaCloud accounts on each plan. It complements the
Pricing Plans document (which describes costs) and the API
Rate Limits document (which describes API throughput limits).

The detailed structured data is in `product_limits.json`. The
narrative below is less precise and is intended for
human-friendly reading.

## NovaCompute limits

NovaCompute workloads are limited along the following axes:

- **CPU**: maximum cores per workload, maximum cores per
  account.
- **Memory**: maximum RAM per workload, maximum RAM per
  account.
- **Disk**: maximum attached storage per workload.
- **Concurrent workloads**: maximum number of workloads that
  can run simultaneously.
- **Bandwidth**: maximum egress per workload per month.

The exact numbers depend on the deployment tier. The
following table summarizes the per-workload ceilings; the
per-account ceilings are typically 10× the per-workload
ceiling. Customers who need higher per-account ceilings
should contact sales.

| Tier        | CPU cores | RAM    | Disk   |
|-------------|-----------|--------|--------|
| Edge        | 2         | 4 GB   | 20 GB  |
| Standard    | 16        | 64 GB  | 500 GB |
| Isolated    | 64        | 256 GB | 2 TB   |

These ceilings are the published maximums. Real-world
performance depends on the underlying instance type, which
is described in the Instance Type Reference.

## NovaDB limits

NovaDB clusters are limited along the following axes:

- **Storage**: maximum database size.
- **Connections**: maximum simultaneous client connections.
- **Replicas**: maximum number of read replicas.
- **Backup retention**: maximum retention period for
  automatic backups.

The exact numbers depend on the cluster tier. The Starter
plan supports up to 100 GB of storage. The Growth plan
supports up to 1 TB. The Business plan supports up to 10 TB.
The Enterprise plan has no storage ceiling.

## NovaStream limits

NovaStream topics are limited along the following axes:

- **Partitions per topic**: maximum number of partitions.
- **Retention**: maximum message retention period.
- **Throughput**: maximum ingress and egress throughput.
- **Consumer groups**: maximum number of consumer groups per
  topic.

The retention maximum is ninety days on the Business plan
and three hundred sixty-five days on the Enterprise plan.
Throughput is measured in MB/s per partition and is throttled
at the partition level.

## Workspace limits

Each NovaCloud account has at least one workspace. The
following limits apply to workspaces:

- Maximum members per workspace: 100 on Business, unlimited
  on Enterprise.
- Maximum environments per workspace: 50 on Business, 500 on
  Enterprise.
- Maximum API tokens per workspace: 1,000 on all plans.

The Starter and Growth plans allow only one workspace per
account.

## Hard ceilings vs soft ceilings

Most limits described in this document are **hard ceilings**:
requests that would exceed the ceiling return a 400 error
with a code indicating the exceeded limit. Some limits are
**soft ceilings**: requests that would exceed the ceiling
are allowed but generate a warning event in the audit log.

The distinction is documented per-limit in the
`product_limits.json` file under the `enforcement` field.

## What happens when a limit is reached

When a hard ceiling is reached, the affected operation fails
with an error. NovaCloud does not automatically upgrade the
customer to a higher plan. The customer must explicitly
upgrade from the billing dashboard.

When a soft ceiling is reached, the operation succeeds but a
warning is emitted. Soft-ceiling warnings are visible in the
audit log and in the customer dashboard.

## Distractor

A 2024 internal design doc proposed removing the per-workspace
API token limit entirely. The proposal was discussed but
ultimately shelved in favor of a higher fixed limit (1,000)
that applies to all plans. Customers who remember the
discussion and expect the limit to have been removed should
consult the current documentation, which is authoritative.