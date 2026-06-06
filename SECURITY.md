# Security Policy

## Scope and status

CreditGuard is a reference implementation, not a production service. There is no deployed instance and no supported release line. The contents of this repository should be treated as illustrative — adapting them for production use is the consumer's responsibility, including the security review that goes with handling applicant-level data.

That said, if you find a vulnerability in the code as published, please report it.

## Supported versions

Only the `main` branch is supported. There are no backported security fixes for older tags or forks.

## Reporting a vulnerability

Please use **GitHub's private security advisory flow** ("Report a vulnerability" under the Security tab) rather than opening a public issue. That keeps the disclosure non-public until a fix is available.

If GitHub Security Advisories is unavailable for your account, email the maintainer with a short description, a reproducer, and (if relevant) the commit hash you were testing against.

## What counts

In scope:

- Code execution / unsafe deserialisation paths in the FastAPI surface (`serving/`).
- Information leakage in API responses (e.g. unintended fields, stack traces).
- Authentication / rate-limiting concerns in the serving layer (current state: **none implemented** — flagged honestly in the project as a reference implementation without authentication; a finding that explains how to exploit the *absence* of these is informative but not a new vulnerability).
- Dependency-graph vulnerabilities surfaced by tooling, where the project pins a vulnerable transitive version.

Out of scope:

- Applicant-data privacy in the Home Credit dataset itself (the dataset is the responsibility of Kaggle and Home Credit; this project only consumes it).
- Findings that depend on a deployment configuration the project does not ship (e.g. a Dockerfile change you made locally).
- Issues in the `governance/` documents — those are advisory text, not security boundaries.

## Disclosure timeline

I will acknowledge a report within 7 days and, where there is a fix, aim to publish it within 30 days of acknowledgement. For complex issues, expect coordination on a longer timeline.
