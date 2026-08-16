<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Authentication and Authorization

This document describes how human users, service accounts, and
machine-to-machine integrations authenticate to NovaCloud. It
covers password requirements, multi-factor authentication, API
token issuance and rotation, and the role-based access control
model used throughout the product.

## User authentication

Human users authenticate to the NovaCloud dashboard at
dashboard.example using an email address and a password.
Passwords must meet the following requirements:

- At least 12 characters long.
- At least one uppercase letter, one lowercase letter, one
  digit, and one symbol.
- Not reused across the last 24 passwords for the same user.
- Not present in the Have I Been Pwned corpus.

Passwords are hashed with Argon2id before storage. NovaCloud
never stores plaintext passwords and cannot recover a forgotten
password on behalf of a user. Forgot-password flows issue a
reset link that is valid for 30 minutes.

## Multi-factor authentication (MFA)

MFA is required for all users with the `owner` role. MFA is
optional but recommended for all other roles. NovaCloud
supports the following MFA methods:

- Time-based one-time passwords (TOTP), compatible with
  Google Authenticator, 1Password, and Authy.
- WebAuthn security keys, including YubiKey and Titan.
- Push notifications through the NovaCloud mobile app.

SMS-based one-time passwords are NOT supported. NovaCloud
retired SMS MFA in March 2024 because of the well-known SIM
swap risk. Customers who still have SMS as their second
factor on file have until December 2026 to enroll a different
method. After that date, accounts will be prompted to enroll
on next login.

## API tokens

API tokens are issued from the dashboard at
dashboard.example/settings/tokens. Each token has a name, a
scope, and an optional expiration date. Tokens can be scoped
to a single project, a single environment, or the entire
account. Tokens are shown to the user exactly once at the time
of creation and cannot be retrieved later.

Tokens issued with an expiration date expire automatically at
the configured time. Tokens without an expiration date do
not expire and must be revoked manually. NovaCloud recommends
that all production tokens have an expiration date of 90 days
or less.

## Token rotation

To rotate a token, create a new token with the same scope,
deploy it to all consuming services, then revoke the old
token. NovaCloud does not have a "rotate in place" operation;
a new token must always be issued.

Service accounts should rotate tokens on a 60-day cadence.
The recommended rotation process is described in the
Developer Documentation document.

## Role-based access control

NovaCloud uses a four-role model:

- **owner**: full control over the account, including
  billing and deletion. There must always be at least one
  owner per account.
- **admin**: full control over resources, but cannot change
  billing or delete the account.
- **developer**: can deploy, configure, and read resources.
  Cannot manage users or billing.
- **viewer**: read-only access to resources.

Custom roles are available on the Business and Enterprise
plans. Custom roles allow customers to define their own
permission sets with arbitrary combinations of the
twenty-three underlying permissions.

## Service accounts

Service accounts are non-human identities used by automated
systems. Service accounts can hold API tokens but cannot log
in to the dashboard with a password. Service accounts count
against the seat count on Business and Enterprise plans and
do not count against the seat count on Starter and Growth plans.

## SSO and SCIM

Single sign-on via SAML 2.0 is available on the Enterprise
plan. NovaCloud supports identity providers including Okta,
Microsoft Entra ID, Google Workspace, and OneLogin. SCIM 2.0
provisioning is supported on Enterprise plans for the same
identity providers.

## Distractor

A frequently-asked question on the support forum asks whether
NovaCloud supports OAuth 2.0 device-code grant. It does not.
Device-code grant is on the roadmap but is not expected
before Q4 2026. Customers who require device-code grant today
should implement their own authorization server and exchange
the resulting authorization code for a NovaCloud API token
using the `/v1/oauth/token` endpoint, which accepts standard
authorization-code exchanges.