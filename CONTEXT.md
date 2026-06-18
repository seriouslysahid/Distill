# CONTEXT.md — AI Coding Assistant Instructions

> **These rules are binding constraints, not suggestions.** They apply to every feature, phase, and module. If you cannot satisfy all four pillars for a given feature, stop and escalate rather than shipping incomplete or unintentional code.

---

## 1. Purpose & Philosophy

This document is the permanent instruction set for any AI coding agent working on this codebase. It is loaded at the start of every session and defines the non-negotiable standards that govern how code is designed, implemented, verified, and delivered.

The core philosophy rests on **four interdependent pillars**:

| Pillar | One-Line Definition |
|--------|-------------------|
| **Modularity** | Every module is a general-purpose building block — no business-specific coupling |
| **Completeness** | No feature ships until it is fully implemented, tested, and verified |
| **Auditability** | Systematic verification catches logic failures, ID mismatches, and integration gaps |
| **Intentionality** | Every line of code has a clear Why and How — no speculative or cargo-cult code |

---

## 2. Modular, Use-Case Agnostic Architecture

### The Principle

Every module, service, component, and data structure must be designed as a **general-purpose building block**. The codebase is a platform, not a product. A dialer module should be equally usable for agentic voice-AI debt collection, human tele-agent marketing, automated surveys, appointment reminders, or any future use case. Features must never be built with assumptions that limit them to a single business scenario.

### Enforcement Rules

- **No business logic in infrastructure** — Core modules (dialer, auth, storage, messaging, scheduling) must contain zero references to specific business domains. If you write `debtAmount` or `campaignName` inside a dialer service, you have violated this rule.
- **Configuration over hardcoding** — Any behavior that could vary across use cases must be driven by configuration, not code. This includes call scripts, retry policies, escalation rules, success criteria, and data schemas.
- **Interface-driven design** — Modules communicate through well-defined interfaces (APIs, event contracts, data schemas), never through shared implementation details.
- **Pluggable strategies** — Where business rules are needed, implement them as swappable strategy patterns. The dialer does not decide what happens after a call; it invokes a configurable post-call handler.
- **Domain-agnostic naming** — Use `contact` instead of `debtor`, `campaign` instead of `collectionDrive`, `interaction` instead of `callAttempt`. Names describe structural concepts, not business applications.

### Anti-Patterns to Avoid

- Hardcoding business-specific thresholds inside shared services
- Embedding workflow state machines inside core modules
- Creating database schemas that assume a single entity type (e.g., `Debtor` table instead of generic `Contact` with extensible attributes)
- Passing business context through infrastructure function signatures (e.g., `dialer.initiateCall(debtId, collectorId)` instead of `dialer.initiateCall(context, handler)`)
- Building UI components that display business-specific labels or assume a particular workflow

---

## 3. No Half Implementations

### The Principle

Every feature is either **fully implemented or not implemented at all**. A half-implemented feature is worse than a missing feature — it creates false progress, introduces hidden bugs, and forces the next developer to reverse-engineer incomplete logic.

### The Completion Checklist

Before marking any feature as done, verify **every item**:

- [x] **Happy path verified** — Primary use case works end-to-end in a clean environment
- [x] **Error handling comprehensive** — Every external call has try/catch with meaningful messages; every API validates inputs; no error path silently swallows exceptions
- [x] **Edge cases addressed** — Empty states, null values, concurrent access, oversized inputs, network failures, timeouts, permission denials — all handled explicitly, no TODO placeholders
- [x] **Tests written and passing** — Unit tests for business logic, integration tests for cross-module interactions, failure-mode tests, all passing in CI
- [x] **Configuration externalized** — All tuneable parameters are configurable via env vars, config files, or DB entries; defaults are sensible and documented; no magic numbers
- [x] **Integration surface clean** — API contracts versioned, DB migrations reversible, event schemas backward-compatible
- [x] **Documentation updated** — README, API docs, and architecture diagrams reflect current state

### The Stop Rule

> If you cannot complete all items in the checklist, **do not ship partially**. Instead:
> 1. Revert incomplete changes to a stable state
> 2. Document what remains in a tracking issue with full context
> 3. Escalate to the project lead with a clear statement of what is missing
>
> A clean revert is always better than a partial merge.

---

## 4. Audit Protocol

### The Principle

Every implementation must be audited before it is considered complete. An audit is a **systematic verification** — not a casual code review. AI-generated code is particularly prone to hallucinated function names, mismatched IDs, orphaned references, and logic that appears correct but fails under specific inputs. These errors often produce silent data corruption rather than immediate failures.

### 4.1 Data Integrity Audit

- **ID consistency** — Every foreign key points to an existing record; IDs generated in one module are correctly consumed in another; no ID collision risk between modules
- **Schema alignment** — DB schemas match application data structures; migrations applied in correct order; no orphaned columns or missing code-side mappings
- **Data flow completeness** — Every entity traceable from creation → processing → storage → retrieval; no silent data drops or unintended transformations; event payloads contain all fields expected by consumers

### 4.2 Logic Correctness Audit

- **Backend logic verification** — Walk every conditional branch; verify each produces the correct result; check for unreachable or contradictory branches
- **State machine integrity** — Every state reachable, every transition valid, no invalid state reachable through any operation sequence; rollback paths exist for every forward transition
- **Concurrency safety** — Every shared mutable resource properly synchronized; no race conditions, deadlocks, or stale reads; idempotency guaranteed for retry-able operations

### 4.3 Integration Audit

- **API contract compliance** — Request/response schemas match documentation; required fields present; error responses follow agreed format
- **Cross-module reference integrity** — Module-to-module calls use correct endpoints, signatures, and formats; no module depends on another's internal implementation details
- **Missing entity check** — Every referenced entity has a corresponding DB table, config entry, or runtime registration; undefined references are guaranteed runtime failures

### 4.4 Security Audit

- **Input validation** — Every external input validated for type, length, format, and range; no user input passed directly to queries, commands, or templates
- **AuthN/AuthZ** — Every sensitive endpoint requires authentication; every restricted action verifies permissions; no auth bypass for convenience
- **Secret management** — No credentials hardcoded, logged, or exposed in errors; all secrets from secure stores at runtime

---

## 5. Intentional Development

### The Principle

No code should exist without a clear, articulable reason. Every function, variable, branch, and architectural decision must answer: **Why does this exist?** and **How does it serve the system's purpose?** If you cannot answer both concisely, the code is unintentional and should be justified or removed.

### The Intentionality Test

Before writing any code, answer these questions. If you cannot, stop and clarify:

1. **Why am I adding this code?** — What specific requirement or constraint does it satisfy? "Best practice" is not sufficient.
2. **How does this interact with existing code?** — What calls this? What does it call? What data does it read/write? If you cannot trace the connection, the code is speculative.
3. **What would break if I removed this?** — If "nothing," the code is dead weight — a maintenance burden and potential bug vector.
4. **Is this the simplest solution?** — Complexity must be earned. Every abstraction, pattern, and generic interface must justify itself by solving a real problem.

### Prohibited Patterns

- **Speculative scaffolding** — No interfaces, abstract classes, or plugin architectures for functionality that does not yet exist. Build for today's requirements with clean extension points.
- **Cargo-cult conventions** — No patterns applied because they are popular. Apply patterns because they solve a specific, demonstrable problem in this codebase.
- **Comment-driven development** — Comments explain **why**, never **what**. If code is not self-explanatory, rewrite the code. A TODO without a tracking issue and deadline is a promise never kept.
- **Defensive over-engineering** — No caching layers, circuit breakers, retry mechanisms, or observability hooks without demonstrated need. Each adds complexity and failure modes.
- **AI placeholder code** — No stub functions or mock implementations that say "implement later." If the implementation is unknown, the design is not ready.

---

## 6. Workflow Enforcement

### The Mandatory Development Sequence

All development must follow this four-phase sequence. Skipping phases or reordering is not permitted.

**Phase 1: Research**
Explore the existing codebase thoroughly. Understand current architecture, established patterns, involved modules, and constraints. Search for similar functionality that already exists. Be completely certain before proceeding.

**Phase 2: Plan**
Create a detailed implementation plan: what files change, what new modules are created, how data flows, what interfaces are exposed, and how the feature integrates. Include reasoning for each decision. Present the plan before implementing. Verify against the four pillars.

**Phase 3: Implement**
Implement the plan precisely — no deviations without updating the plan first and verifying the deviation satisfies all pillars. Write tests alongside implementation. Run the full audit checklist after implementation. Fix issues before proceeding.

**Phase 4: Verify & Commit**
Run the complete test suite. Verify no regressions. Confirm implementation matches the plan. Run the audit checklist one final time. Commit with a descriptive message referencing the plan and noting any deviations.

### Context Management

- **Re-read this file periodically** — During long sessions, explicitly re-read this document every few messages. Context window drift is real.
- **State which rules you are applying** — When responding to requests, briefly note which rules you are following. This makes compliance visible.
- **Start fresh for new tasks** — When switching to a significantly different task, start a new session. Stale context degrades quality.
- **Challenge assumptions** — If a request conflicts with these principles, say so directly. Do not silently comply with violations. Explain the conflict and propose an alternative.

---

## 7. Quick Reference Card

### Red Flags — Stop Immediately If You Are About To:

- Add a business-specific variable name to a shared module
- Write a TODO comment instead of implementing the logic
- Create an abstract interface for functionality that does not yet exist
- Hardcode a configuration value that could vary across deployments
- Skip the audit because "it's a small change"
- Implement something without being able to explain why it is needed
- Ship a feature that fails any item on the completion checklist
- Deviate from the plan without updating the plan first
