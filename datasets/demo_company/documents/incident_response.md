<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Incident Response and Status

This document describes how NovaCloud detects, responds to,
and communicates about service incidents. It also describes
how customers can subscribe to status updates and how to file
an incident-related support ticket.

## Status page

The public status page is hosted at status.example and is
the authoritative source of real-time service status. The
page is updated automatically from the same alerting system
that pages the on-call engineers.

The status page shows status for the following components:

- NovaCompute control plane (all regions).
- NovaCompute data plane (each region separately).
- NovaDB (each region separately).
- NovaStream (each region separately).
- NovaObserve (global).
- NovaShield (global).
- NovaEdge (global).
- The dashboard and API.

Each component has its own status indicator and its own
incident history.

## Incident severity levels

Incidents are classified into the same four severity levels
as support issues, with slightly different definitions:

- **Severity 1**: a major service is down for more than 10%
  of customers OR a major service is completely down for any
  customer.
- **Severity 2**: a major service is impaired but operational.
- **Severity 3**: a minor service is impaired.
- **Severity 4**: cosmetic or informational.

The definitions above apply to NovaCloud's incidents, not
to customers' own incidents.

## Communication channels

During a Severity 1 or Severity 2 incident, NovaCloud
communicates through the following channels:

- The status page at status.example.
- A live blog embedded on the status page.
- The `@NovaCloudStatus` Twitter/X account.
- Email to all subscribers of the status updates mailing
  list.

The mailing list is opt-in. Customers can subscribe from
the dashboard.

## Postmortems

NovaCloud publishes a public postmortem for every Severity 1
incident within fourteen days of resolution. The postmortem
includes the timeline, the contributing factors, and the
remediation actions. Remediation actions are tracked in a
public backlog at status.example/backlog.

Postmortems for Severity 2 incidents are published on
request. Postmortems for Severity 3 and Severity 4 incidents
are not published by default.

## Customer-side incidents

If a customer experiences an incident that they believe is
caused by NovaCloud, they should:

1. Open a Severity 1 or Severity 2 support ticket, depending
   on impact.
2. Reference the relevant component from the status page, if
   applicable.
3. Provide logs, traces, and any other supporting evidence.

NovaCloud's incident response team will investigate and may
request additional information. For confirmed customer-impacting
incidents, the customer is eligible for the SLA credits
described in the SLA Reference.

## Customer-initiated incident reports

Customers who believe their account has been compromised
should contact the security team directly at
security@example rather than opening a support ticket. The
security team has a separate pager rotation and can engage
the forensics team if needed.

## Internal coordination

The NovaCloud incident response team coordinates with the
following internal teams during a Severity 1 incident:

- Platform engineering.
- Customer engineering.
- Communications.
- Legal (if the incident involves customer data).
- Executive on-call.

The executive on-call has the authority to declare an incident
"major" and trigger additional communication requirements.

## Distractor

A 2022 incident was internally nicknamed "the Tuesday outage"
even though it actually occurred on a Wednesday. The
nickname persists in internal documentation. Customers
reading older internal postmortems may be confused by the
discrepancy. The official incident ID was `INC-2022-0817` and
the correct date is 17 August 2022.