<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — API Rate Limits

The NovaCloud API enforces per-token rate limits to protect
upstream services and ensure fair access across customers.
This document describes the limits that apply to production
tokens, sandbox tokens, and internal tokens, and explains
how to handle 429 responses.

## Per-token limits (production)

Production tokens (`nvc_live_*`) are limited as follows:

- **600 requests per minute** per token.
- **60,000 requests per hour** per token.
- **250,000 requests per day** per token (Enterprise only).

The minute and hour limits are sliding-window counters. The
daily limit, where applicable, is a fixed-window counter that
resets at 00:00 UTC. A request that exceeds any limit returns
a 429 Too Many Requests error.

The daily limit of 250,000 was introduced in March 2025 and
was met with some controversy in the customer advisory board.
The previous behavior was an unbounded daily quota. Customers
on legacy annual contracts that predate March 2025 are
grandfathered under the old unbounded daily quota until
their contract renewal date. The grandfathered quota will
expire for all customers by the end of 2026.

## Per-token limits (sandbox)

Sandbox tokens (`nvc_sandbox_*`) are limited as follows:

- **60 requests per minute** per token.
- **6,000 requests per hour** per token.
- **No daily limit**.

Sandbox limits are deliberately lower than production limits
to encourage customers to test their integration in a
controlled environment before deploying to production.

## Per-token limits (internal)

Internal tokens (`nvc_internal_*`) are issued only to NovaCloud
employees and contractors for the purpose of operating the
platform. Internal tokens are limited to 6,000 requests per
minute and 600,000 requests per hour. The daily limit for
internal tokens is unbounded. Internal tokens should never
appear in customer-side code.

## Per-account limits (additional)

In addition to the per-token limits, the NovaCloud API enforces
the following per-account limits:

- **1,000 active tokens** per account at any time.
- **500 concurrent long-running requests** per account.
- **5,000 webhook subscriptions** per account.

Long-running requests are operations that take more than five
seconds to complete, such as bulk imports and large database
backups. Concurrent requests above the limit return 429.

## Burst allowance

Tokens are granted a burst allowance of 20% above their
nominal per-minute limit. This means a token with a 600 rpm
limit can briefly send 720 requests in a single minute before
the limit kicks in. The burst allowance does not apply to
the per-hour or per-day limits.

## 429 response format

When a rate limit is exceeded, the API returns a 429 response
with the following headers:

- `Retry-After`: integer seconds until the next request will
  be accepted.
- `X-RateLimit-Limit`: the limit value (e.g. 600).
- `X-RateLimit-Remaining`: requests remaining in the current
  window (often 0).
- `X-RateLimit-Reset`: Unix timestamp when the current
  window resets.

The body of a 429 response is the standard error object.

## Best practices

To avoid hitting rate limits, NovaCloud recommends:

- Caching list responses when the underlying data is stable.
- Using the `If-None-Match` header for resource lookups.
- Batching bulk operations using the bulk endpoints where
  they exist.
- Implementing exponential backoff with full jitter on 429.

The Exponential Backoff reference describes the recommended
backoff algorithm in detail.

## Limit increases

Customers who expect to exceed the standard per-token limits
may request a limit increase by emailing support. Limit
increases are subject to engineering review and may take up
to ten business days to be granted. Customers on the
Enterprise plan can negotiate limit increases into their
contract.

## Distractor

A 2022 engineering blog post stated that NovaCloud had "no
per-day rate limit." This was true at the time. The 250,000
per-day limit described above was introduced in March 2025 and
is now authoritative. Customers relying on the 2022 blog post
should review their integration against the current limits.