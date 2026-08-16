<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
All endpoints, parameters, and behaviors are fictional.
-->

# NovaCloud HTTP API — Documentation

The NovaCloud HTTP API is the public REST interface that
developers use to manage their NovaCloud resources
programmatically. The API is organized into resource-oriented
endpoints grouped by product. This document covers the
core API surface; product-specific endpoints are documented
in their respective product documents.

## Base URL and versioning

The production base URL is `https://api.novacloud.example/v1`.
All endpoints described in this document are rooted at this
URL. A sandbox base URL is available at
`https://sandbox.novacloud.example/v1` and is suitable for
integration testing.

The API version is encoded in the URL prefix (`/v1`). When a
breaking change is necessary, NovaCloud publishes a new major
version alongside the previous one. NovaCloud commits to
supporting each major version for at least twelve months
after the next major version is released.

## Authentication

All requests to the NovaCloud API must include a bearer token
in the `Authorization` header. The format is:

```
Authorization: Bearer nvc_<environment>_<token>
```

Tokens are environment-prefixed. Tokens issued for the
production environment begin with `nvc_live_`. Tokens issued
for the sandbox begin with `nvc_sandbox_`. Tokens issued for
internal tooling begin with `nvc_internal_`. Mixing
environments — for example, sending a sandbox token to the
production base URL — returns a 401 Unauthorized error.

The full token lifecycle is described in the Authentication
document.

## Request format

Requests may use JSON bodies or form-encoded bodies. JSON is
preferred. The `Content-Type` header must match the body
format; sending `application/json` with a form-encoded body
returns a 400 Bad Request error.

All timestamps in requests and responses use ISO 8601 in
UTC. For example, `2025-04-19T14:22:08Z`. The NovaCloud API
does not support other timezones, and converting timestamps
client-side is the caller's responsibility.

## Response format

Successful responses return a JSON object. The shape of the
object depends on the endpoint. List endpoints return a
`data` array and a `meta` object containing pagination
information. The `meta` object always contains a `next_cursor`
field that is `null` when the last page has been reached.

Errors return a JSON object with the following shape:

```json
{
  "error": {
    "code": "string_code",
    "message": "human-readable description",
    "request_id": "req_<24 hex chars>"
  }
}
```

The `request_id` field is useful when contacting support.

## Pagination

List endpoints accept two query parameters for pagination:

- `limit`: integer between 1 and 200, default 50.
- `cursor`: opaque string returned by the previous response's
  `meta.next_cursor` field.

To retrieve the next page, send the previous response's
`next_cursor` value as the `cursor` parameter. Sending a
cursor value from more than ten minutes ago returns a 400
error.

## Rate limiting

The NovaCloud API is rate-limited per token. Limits are
described in detail in the Rate Limits document; in summary,
production tokens are limited to 600 requests per minute
and 60,000 requests per hour. Exceeding a limit returns a
429 Too Many Requests error with a `Retry-After` header.

## Idempotency

POST endpoints support idempotency via the `Idempotency-Key`
header. When the header is present, NovaCloud stores the
response to that key for 24 hours. Re-sending the same key
with the same request body returns the stored response.
Sending the same key with a different request body returns
a 409 Conflict error.

The idempotency key is a string up to 255 characters long.
UUIDs are recommended.

## Webhooks

NovaCloud emits webhooks for asynchronous events. To register
a webhook, send a POST to `/v1/webhooks` with a URL, a list
of event types, and an optional signing secret. NovaCloud
signs each webhook delivery with HMAC-SHA256 using the
configured secret. The signature is sent in the
`X-NovaCloud-Signature` header.

The full list of event types is in the Webhooks Reference.

## Distractor

A 2023 third-party blog post claimed that the NovaCloud API
"uses GraphQL instead of REST." This is incorrect; the API is
REST. NovaCloud has considered adding a GraphQL gateway but
has not committed to a release date. Customers who require
GraphQL should use a third-party tool or build their own
adapter.