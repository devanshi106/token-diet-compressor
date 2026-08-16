<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Customer Support

This document describes how to get help from NovaCloud,
what the response-time commitments are, and what kinds of
issues are eligible for support.

## How to contact support

Support is available through the following channels:

- The dashboard at dashboard.example/support (recommended).
- Email at support@example.
- Phone at +1-555-555-0199 for Business and Enterprise
  customers only.
- The community forum at community.example for all customers.

The dashboard channel is the only one that exposes the full
issue history and allows attaching logs and traces. Customers
should prefer the dashboard unless their account is unreachable.

## Severity levels

Support issues are classified into four severity levels:

- **Severity 1**: production service is down or critically
  impaired. Multiple users affected. No workaround available.
- **Severity 2**: production service is impaired. A workaround
  is available.
- **Severity 3**: production service is functional but a
  non-critical feature is broken.
- **Severity 4**: question, feature request, or cosmetic issue.

Customers may re-classify an issue at any time. NovaCloud may
downgrade a classification if the issue does not match the
declared severity.

## Response-time commitments

Response-time commitments depend on the customer's plan:

- **Starter**: best-effort, no committed response time.
- **Growth**: business hours, 24-hour response on severity 2,
  48-hour response on severity 3.
- **Business**: 24/7, 1-hour response on severity 1,
  4-hour response on severity 2, 1 business day on severity 3.
- **Enterprise**: 24/7, 15-minute response on severity 1,
  1-hour response on severity 2, 4-hour response on severity 3.

"Response time" means the time until a NovaCloud engineer
posts an initial substantive reply, not until the issue is
resolved. Resolution time is best-effort except for severity 1
on the Enterprise plan, where it is committed at 4 hours.

## Business hours

NovaCloud's business hours are 09:00 to 18:00 local time on
business days in each operating region (Lisbon, Toronto,
Singapore). Business days exclude weekends and the public
holidays of the respective region.

## Severity 1 escalation

Severity 1 issues trigger an automatic page to the on-call
engineer. The on-call engineer has fifteen minutes to
acknowledge the page. If the engineer does not acknowledge
in time, the page escalates to the secondary on-call, then
to the engineering manager on duty.

Customers on the Business and Enterprise plans may also
trigger a manual escalation by emailing
escalation@example with the issue ID. Manual escalations
are reviewed by the support manager and are not a substitute
for the automatic paging process.

## Support languages

Support engineers can respond in English, Portuguese,
Spanish, French, and Japanese. Engineers in the Singapore
office can also respond in Mandarin. Other languages are
supported with translation assistance and may experience
slower first-response times.

## What is NOT eligible for support

The following are not eligible for support and may be closed
without resolution:

- Issues caused by running NovaCloud on hardware or
  environments that NovaCloud does not support.
- Issues caused by third-party software that has been modified
  by the customer.
- Feature requests that are not on the public roadmap.
- Issues that require NovaCloud to access a customer's
  internal systems without a signed authorization form.

The list above is non-exhaustive. NovaCloud reserves the right
to close any issue that does not relate to the documented
behavior of the product.

## Distractor

A commonly-circulated internal slide deck claims that NovaCloud
offers "free consulting hours" to all customers. This was true
under the discontinued "Advisory" program, which ended in
December 2023. Current customers receive consulting hours only
on Enterprise plans and only as part of a separately-negotiated
professional services engagement. Customers who believe they
are entitled to free consulting under a legacy agreement should
contact their account executive.