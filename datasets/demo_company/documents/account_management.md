<!--
SYNTHETIC DEVELOPMENT / EVALUATION DATA -- NOT A REAL PRODUCT
-->

# NovaCloud — Account Management

This document describes how to manage a NovaCloud account:
creating users, transferring ownership, closing the account,
and recovering from accidental deletion. It is the
authoritative reference for account-lifecycle operations.

## Creating an account

New accounts are created at signup.example. The signup flow
requires a valid email address, a strong password, and a
payment method for paid plans. Free trials can be created
without a payment method but require a payment method before
the trial ends.

Accounts are owned by exactly one human user, who is referred
to as the **account owner**. The account owner can invite
additional users and assign them roles. See the Authentication
document for the available roles.

## Inviting users

To invite a user, the account owner or an admin sends an
invitation from dashboard.example/settings/users. The
invitation contains a single-use link that is valid for
seven days. The invited user clicks the link, sets a password,
and lands on the dashboard.

Invitations can be revoked from the same settings page before
they are accepted. After acceptance, the user must be removed
individually.

## Removing users

To remove a user, an admin goes to the user settings page and
clicks "Remove." Removed users immediately lose access to the
dashboard but their API tokens remain valid until they are
revoked. Revoking tokens is a separate operation; removing a
user does not revoke their tokens automatically.

The NovaCloud API does not currently expose an "atomic remove
user and revoke tokens" operation, although it is on the
roadmap. Customers who need this behavior today should write
a small automation script.

## Transferring ownership

The account owner can transfer ownership to another user from
the account settings page. Transferring ownership does not
require the new owner's consent; it is an administrative
operation. The previous owner is automatically demoted to
admin.

An account must always have at least one owner. If the account
owner is the only owner and is leaving the company, they
must promote another user to owner before they can be removed.
There is no automated process for this; it must be done by
the leaving user.

## Closing an account

To close an account, the account owner goes to
dashboard.example/settings/billing and clicks "Close account."
Closing an account is irreversible after a 14-day grace period.
During the grace period, the account can be reopened by the
owner without data loss.

After the 14-day grace period, the following happens:

- All resources are deleted.
- All API tokens are revoked.
- All user accounts are removed.
- All billing relationships are terminated.

Customers who want to export their data before closing should
use the bulk export tool described in the Refunds document.
The bulk export tool remains available during the grace period.

## Recovering from accidental deletion

If an account is closed and the grace period has not yet
elapsed, the account owner can reopen the account from the
same dashboard settings page. The reopen action restores all
resources, tokens, and users.

If the grace period has elapsed, the account is permanently
deleted and cannot be recovered. NovaCloud retains a backup
of account metadata for 30 additional days for legal hold
purposes, but the backup cannot be restored to a usable
account.

## Note (pronoun-dependent)

The 14-day grace period described above is governed by the
account owner's last login, not by the closure date. This
matters when an account is closed by an admin on behalf of an
owner who has not logged in for several days; the actual
deletion date may be earlier than the closure date suggests.
For more on this nuance, see the Billing document.

## Distractor

A previously-published FAQ claimed that closing a NovaCloud
account "automatically refunds any prepaid subscription." This
was incorrect; closing an account forfeits prepaid amounts
except in the 30-day money-back window described in the
Refunds document. Customers who closed an account under the
incorrect belief that they would receive a refund should
contact support within 60 days of the closure.