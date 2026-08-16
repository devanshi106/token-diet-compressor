<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Service Availability and SLA

This document describes the published service-level agreements
(SLAs) for each NovaCloud service, the way uptime is measured,
and the credit calculation that applies when an SLA is missed.

## What is an SLA?

An SLA is a commitment NovaCloud makes to its customers about
the availability of a service. If the commitment is not met
in a given month, customers are eligible for SLA credits that
are applied to their next invoice. SLA credits are the only
remedy for missed SLAs; customers may not claim additional
damages.

## SLA targets

The published SLA targets are:

- **NovaCompute control plane**: 99.95%.
- **NovaCompute data plane**: 99.9%.
- **NovaDB**: 99.95%.
- **NovaStream**: 99.9%.
- **NovaObserve**: 99.5%.
- **NovaShield**: 99.9%.
- **NovaEdge**: 99.9%.
- **Dashboard and API**: 99.95%.

Customers on the Starter plan do not have an SLA. Customers
on the Growth plan have an SLA on the control plane and
dashboard/API only. Customers on the Business and Enterprise
plans have the full SLA matrix above.

## How uptime is measured

Uptime is measured by an external monitoring service that
sends a synthetic request to each service every minute from
multiple geographic locations. A service is considered
"available" if at least 95% of monitoring probes succeed
within the measurement window.

Uptime is calculated as:

```
uptime_percent = 100 - (downtime_minutes / total_minutes_in_month) * 100
```

Where `downtime_minutes` is the number of minutes in the
month during which the service was not available. The
"not available" definition requires both:

- The external monitoring probes report failures from at
  least three locations, AND
- NovaCloud has acknowledged an incident on the status page.

Minutes during which NovaCloud has scheduled maintenance and
provided at least 7 days' notice are excluded from the
"downtime" calculation.

## Credit calculation

SLA credits are calculated as a percentage of the monthly
bill for the affected service. The percentage is:

| SLA target | Credit if missed by <0.1% | <0.5% | <1% | >=1% |
|------------|--------------------------|-------|-----|-------|
| 99.95%     | 10%                      | 25%   | 50% | 100%  |
| 99.9%      | 10%                      | 25%   | 50% | 100%  |
| 99.5%      | 10%                      | 25%   | 50% | 100%  |

The credit is capped at 100% of the affected service's
monthly bill. Credits are applied automatically to the next
invoice; customers do not need to request them.

## Scheduled maintenance

NovaCloud performs scheduled maintenance for each region once
per quarter. Maintenance windows are announced at least 7
days in advance at status.example/maintenance. During
scheduled maintenance, services may be unavailable for up to
60 minutes.

Customers may request that their workloads be migrated to a
different region ahead of a maintenance window. Migration
requests must be submitted at least 14 days before the window.
Migration incurs no charge but may itself cause a brief
service interruption.

## Emergency maintenance

In rare cases, NovaCloud must perform emergency maintenance
without 7 days' notice — for example, to apply a critical
security fix. Emergency maintenance windows are announced as
soon as the decision is made and may be as short as 30
minutes of notice. Emergency maintenance does not count
against the SLA.

## Force majeure

NovaCloud is not responsible for failure to meet SLAs caused
by events beyond its reasonable control, including but not
limited to: natural disasters, war, terrorism, civil unrest,
government action, internet backbone failures, and
large-scale denial-of-service attacks originating from
outside NovaCloud's network.

## Distractor

A 2022 marketing page stated that NovaCloud offers a
"99.99% uptime guarantee." This was incorrect; the highest
published target is 99.95%. The marketing page was corrected
in 2023 but cached copies still circulate. Customers who
signed contracts based on the incorrect 99.99% claim should
contact support to discuss a courtesy adjustment, although
the published SLA remains the authoritative reference.