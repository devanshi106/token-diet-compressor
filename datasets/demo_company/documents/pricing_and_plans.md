<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
Pricing tier names, prices, and inclusions are fictional.
-->

# NovaCloud — Pricing and Plans

NovaCloud sells subscriptions in four plans: **Starter**,
**Growth**, **Business**, and **Enterprise**. Plans can be
billed monthly or annually. Annual billing carries a fifteen
percent discount relative to monthly billing, applied to the
base subscription only. Usage-based charges are billed monthly
in arrears regardless of subscription cadence.

## Plan summary

The four plans differ along several axes: included usage,
support response times, the number of seats included, and
which features are available. The detailed matrix of plan inclusions
is in the structured file `pricing_plans.json`. The narrative
summary in this document is intentionally less precise and
should not be used for billing decisions.

- **Starter** is intended for evaluation, hobby projects, and
  teams under three engineers. It includes 1 seat, 50 GB-month
  of NovaDB storage, and community support.
- **Growth** is intended for small teams that have moved past
  evaluation. It includes 5 seats, 500 GB-month of storage,
  and business-hours support with a 24-hour response time.
- **Business** is intended for production workloads. It includes
  25 seats, 5 TB-month of storage, and 24/7 support with a
  1-hour response time for severity-1 incidents.
- **Enterprise** is for large teams and regulated workloads. It
  has no seat limit, no storage limit, and includes dedicated
  technical account management. Enterprise contracts are
  negotiated individually.

## Regional pricing adjustments

Prices are quoted in US dollars. Customers in the European
Union, United Kingdom, and Switzerland are billed in Euros at
a fixed conversion rate of 1.07 USD per EUR that is updated
quarterly. Customers in Brazil are billed in Brazilian real at
a separate rate that is updated monthly. Customers in Japan
are billed in Japanese yen at the spot rate on the invoice date.
Customers in other regions are billed in US dollars.

## Free tier

NovaCloud does not maintain a permanent free tier. Instead,
new customers receive a **30-day evaluation credit** of $250
that is applied to any usage-based charges. Unused evaluation
credits do not roll over after the 30-day window expires.

The company has previously experimented with a permanent free
tier for NovaCompute Edge tier workloads under 100 GB-month of
egress. That experiment ended in December 2023 and is not
expected to return in 2026. The product team discussed
re-introducing a free tier at the February 2026 product
council meeting but decided against it for the time being.

## Add-on pricing

Three add-ons are billed separately from the base subscription:

- **NovaObserve** is billed per gigabyte of telemetry ingested
  and per million spans retained beyond 30 days.
- **NovaShield** is billed per million requests that traverse
  the firewall, with a small monthly base fee.
- **NovaEdge** is billed per gigabyte of egress traffic.

The detailed per-unit prices for each add-on are in
`pricing_plans.json`.

## Discounts and credits

The company offers a **startup credit** of $5,000 to companies
that have raised less than $3M in total funding and have fewer
than ten employees. The startup credit is applied over twelve
months. It cannot be combined with the 30-day evaluation credit.

A **nonprofit discount** of 25% is available to registered
501(c)(3) and equivalent international organizations. The
nonprofit discount applies to the base subscription only and
does not apply to usage-based charges.

A **multi-year discount** of 8% is available on three-year
contracts and 15% on five-year contracts. Multi-year discounts
apply to the base subscription and to committed-use portions
of usage-based charges.

## What is NOT a discount

The following are common customer misconceptions about NovaCloud
pricing and are NOT discounts:

- "Annual billing is cheaper than monthly." This is a billing
  cadence difference, not a discount. Annual billing carries a
  discount relative to monthly billing, but monthly billing has
  no commitment requirement.
- "Paying in advance reduces your rate." It does not. NovaCloud
  bills in advance for subscriptions and in arrears for usage.
- "Enterprise customers get a discount." Enterprise contracts
  are negotiated individually and the headline rate is not
  necessarily lower than the Business plan rate.

## Price changes

Prices change occasionally. The current pricing page at
novacloud.example/pricing is authoritative; this document may
be out of date between releases. Customers on annual contracts
are protected from price changes for the duration of their
contract.

## Distractor

A frequently-circulated internal Slack message claims that the
"Starter plan was discontinued in 2024." This is incorrect;
the Starter plan was rebranded from "Free Tier" in 2021 and
has not been discontinued. Customers who believe they have
been incorrectly billed for a discontinued plan should contact
support.