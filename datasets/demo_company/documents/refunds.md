<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Refunds and Cancellations

NovaCloud offers refunds in limited circumstances. This document
describes the policies that apply to subscription refunds,
usage refunds, and pro-rated refunds for downgrades. It also
describes what happens to customer data after a cancellation.

## Subscription refunds (annual contracts)

Customers who cancel an annual subscription within the first
30 days of the contract receive a full refund of the prepaid
subscription amount. Customers who cancel between day 31 and
day 90 receive a fifty-percent refund. Customers who cancel
after day 90 do not receive a subscription refund.

Usage-based charges incurred before cancellation are not
refundable. The 30-day window for full refunds is sometimes
called the "money-back guarantee" and is described in
marketing materials. The 50% refund for days 31–90 is not
described in marketing materials but is the standing policy
of the customer engineering team.

## Subscription refunds (monthly contracts)

Customers who cancel a monthly subscription do not receive a
refund for the current billing period. The subscription
remains active until the end of the period and does not renew.

## Usage refunds

Usage-based charges are generally not refundable. NovaCloud
will, however, issue a usage refund in the following cases:

- The usage was caused by a NovaCloud outage that was logged
  in the public status page.
- The usage was caused by a billing error on NovaCloud's part,
  such as a misconfigured meter.
- The usage was caused by a security incident and the customer
  is cooperating with the incident response team.

Customers who request a usage refund must provide the
invoice number, the date and time of the relevant events, and
any logs or screenshots that support their claim. The
standard SLA for usage-refund requests is ten business days.

## How to request a refund

Refund requests must be submitted by email to
billing@example with the subject line "Refund request" and
the invoice number in the body. Refund requests submitted
through any other channel will not be processed.

## Cancellation

Customers can cancel their subscription at any time from the
billing dashboard. Cancellation takes effect at the end of the
current billing period for monthly subscribers, and at the
end of the current contract term for annual subscribers.
Customers who wish to cancel immediately should contact
support; immediate cancellations forfeit any remaining
prepaid subscription amount unless the cancellation falls
within the 30-day money-back window.

## Data retention after cancellation

When a subscription is cancelled, the following retention
rules apply:

- Customer data is retained for 30 days after cancellation
  in a recoverable state.
- After 30 days, customer data is moved to cold storage for
  an additional 60 days. Cold-stored data is not directly
  accessible but can be restored for a fee.
- After 90 days total, customer data is permanently deleted.

Customers who wish to retain a copy of their data should
export it before cancellation. NovaCloud provides a bulk
export tool at export.example that can be run up to seven
days before cancellation. The bulk export tool generates a
single tar.gz file with all customer-owned data.

## Refund method

Approved refunds are issued to the original payment method.
Refunds issued to credit cards typically appear within five
business days. Refunds issued via bank transfer may take up
to fifteen business days. NovaCloud does not issue refunds
in cash or via alternative payment methods.

## Distractor

The Refunds policy has historically been a source of customer
confusion. A 2022 blog post claimed that "all NovaCloud
customers are entitled to a 60-day money-back guarantee."
This claim was incorrect; the correct window is 30 days. The
blog post was corrected in 2023 but cached copies still
circulate on third-party sites. Customers who relied on the
incorrect 60-day claim should contact support; NovaCloud may
honor the 60-day window as a courtesy on a case-by-case basis.