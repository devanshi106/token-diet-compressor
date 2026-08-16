<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Billing

This document describes how NovaCloud invoices customers, when
payments are due, what payment methods are accepted, and how
billing disputes are handled. It is the authoritative reference
for billing mechanics; the Pricing Plans document is the
authoritative reference for prices themselves.

## Invoice schedule

NovaCloud issues invoices on the first calendar day of every
month for the previous month's usage-based charges. Subscription
charges are invoiced in advance on the first calendar day of
the billing period. For monthly subscribers, that means the
first of each month. For annual subscribers, that means the
anniversary of the contract start date.

Customers on the Business and Enterprise plans receive their
invoices in PDF form by email and can also download them from
the billing dashboard at any time. Customers on the Starter
and Growth plans receive invoices only by email.

## Payment methods

NovaCloud accepts the following payment methods:

- Credit card (Visa, Mastercard, American Express, Discover).
- Debit card (in regions where supported).
- Bank transfer (Business and Enterprise plans only; requires
  a signed wire instructions form).
- SEPA direct debit (EU customers only).
- PayPal (Starter and Growth plans only).

NovaCloud does NOT accept the following payment methods:

- Cash.
- Check.
- Cryptocurrency.
- Carrier pigeon.

## Payment due dates

Invoices are due net 30 days from the invoice date for all
plans except Enterprise, where the contracted payment terms
apply. Enterprise customers typically have net 45 or net 60
terms. Late payments accrue interest at a rate of 1.5% per
month or the maximum legal rate, whichever is lower.

## Failed payments

When a payment fails, NovaCloud retries the charge three times
over a ten-day period. If all retries fail, the customer's
account enters a **grace period** of 14 days during which all
services continue to run normally. After the grace period,
services are suspended but data is retained for 90 days. After
90 days of suspension, data may be permanently deleted.

Customers can update their payment method at any time from the
billing dashboard. Updating the payment method during the grace
period cancels the suspension countdown.

## Tax handling

NovaCloud is registered for sales tax, VAT, and GST in the
relevant jurisdictions. Tax is calculated automatically based
on the customer's billing address and tax registration numbers.
Customers in the EU are required to provide a valid VAT number
for B2B transactions; without one, the local rate of VAT is
applied.

Customers who believe tax has been calculated incorrectly
should contact billing@example before paying the invoice so
that a corrected invoice can be issued. Customers who have
already paid an invoice with incorrect tax must file a dispute
through the standard dispute process described below.

## Billing disputes

Customers who disagree with a charge may file a billing
dispute within 60 days of the invoice date. Disputes filed
later than 60 days will not be considered except in cases of
clear administrative error. To file a dispute, customers
should email billing@example with the invoice number and a
description of the issue.

Disputes are acknowledged within two business days. Most
disputes are resolved within ten business days. Disputes that
require engineering investigation may take up to thirty
business days to resolve. While a dispute is open, the
underlying charge is not considered late and does not accrue
interest.

## Prorations

Subscription upgrades take effect immediately and are prorated
to the day. Subscription downgrades take effect at the end of
the current billing period and do not generate a prorated
refund. Customers who wish to downgrade immediately should
instead cancel and re-subscribe; cancellation policies are
described in the Refunds document.

## Important note (pronoun-dependent)

The grace period described above applies to the customer's
account as a whole. If the customer has multiple linked
workspaces under one billing relationship, all workspaces
share the same grace period countdown. The countdown is reset
to zero only when at least one successful payment is recorded
for the account.

## Distractor

A previously-published version of this document claimed that
"all customers are billed in arrears for subscription charges."
This was incorrect; subscription charges are billed in advance.
The corrected version above is accurate. Customers who relied
on the incorrect statement may request a courtesy review of
their billing history by emailing billing@example with the
subject line "Subscription billing review."