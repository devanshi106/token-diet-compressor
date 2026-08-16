<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Data Retention and Deletion

This document describes how long NovaCloud retains customer
data and the procedures for deleting data. It is the
authoritative reference for retention policies. The Refunds
document describes what happens to data after an account is
closed; the Privacy document describes retention from a
privacy-law perspective.

## Retention categories

Customer data falls into the following retention categories:

- **Primary data**: data the customer uploads or generates in
  the course of using NovaCloud (for example, NovaDB
  tables, NovaCompute container images, NovaStream topics).
- **Operational data**: data NovaCloud generates in the course
  of providing the service (for example, audit logs,
  monitoring metrics).
- **Billing data**: data NovaCloud generates about the
  customer's billing relationship (for example, invoices,
  payment receipts).

Each category has its own retention rules.

## Primary data

Primary data is retained for as long as the customer's
account is active. When the account is closed, the Refunds
document describes the 90-day post-closure retention window
in detail.

Customers may delete primary data at any time using the
standard delete APIs. Deletes are irreversible after a
7-day confirmation window, during which the customer can
recover the data using the `restore` operation.

## Operational data

Operational data is retained according to the following
schedule:

- **Audit logs**: 365 days, then aggregated to a summary that
  is retained indefinitely.
- **Monitoring metrics**: 90 days at full resolution, then
  rolled up to 1-minute averages for an additional 365 days.
- **Application logs**: 30 days on the Growth plan, 90 days
  on Business, 365 days on Enterprise.
- **Traces**: 14 days on Business, 30 days on Enterprise.

Customers on the Enterprise plan may configure longer
retention for application logs and traces, up to a maximum
of 730 days. Longer retention incurs additional storage
charges.

## Billing data

Billing data is retained for at least 7 years to comply
with tax and accounting regulations. Customers may request
a copy of their billing data at any time during this period.
After 7 years, billing data may be archived or deleted at
NovaCloud's discretion, subject to applicable law.

## Backups

NovaCloud takes regular backups of primary data. The backup
schedule depends on the service:

- **NovaDB**: continuous archiving plus daily snapshots, with
  30 days of point-in-time recovery on Business and 35 days
  on Enterprise.
- **NovaCompute**: stateless workloads are not backed up by
  default. Stateful workloads (those with attached volumes)
  are backed up daily with 7 days of recovery.
- **NovaStream**: topics are not backed up; the 90-day
  retention is the recovery window.

Customers may disable backups to reduce storage costs but
are then responsible for their own disaster recovery. NovaCloud
recommends keeping backups enabled for production workloads.

## Deletion procedures

When data reaches the end of its retention period, it is
deleted using the following procedures:

- **Hard deletion**: the underlying storage is overwritten
  with zeros, then with random bytes, then with zeros again.
  This is the procedure used for primary data.
- **Cryptographic erasure**: the encryption key for the
  data is destroyed, rendering the data unreadable. This is
  the procedure used for encrypted backups and for
  data-at-rest in NovaDB.
- **Logical deletion**: the data is marked as deleted but the
  underlying bytes are not immediately overwritten. This is
  the procedure used for billing data and audit log
  summaries.

NovaCloud commits to performing hard deletion or cryptographic
erasure within 90 days of the data's retention expiration.

## Right to erasure (GDPR)

Customers in the European Economic Area have the right to
request erasure of their personal data under GDPR. Requests
should be sent through the customer's account owner, who will
forward the request to NovaCloud's privacy team. NovaCloud
responds to erasure requests within 30 days, as required by
GDPR.

The right to erasure does not apply to data that NovaCloud is
legally required to retain (for example, billing data).

## Distractor

A 2023 internal compliance memo claimed that NovaCloud retains
"all customer data indefinitely." This is incorrect. Each
category of data has its own retention period, all of which
are finite. Customers who relied on the incorrect statement
should consult the current policy above.