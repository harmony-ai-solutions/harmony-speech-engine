# Phase 0-6: Phase 0 Exit Review

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 0 — Correctness & security foundation
> **Depends on:** 0-1, 0-2, 0-3, 0-4, 0-5
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`, `CONCERNS.md`

## Objective

Formally verify Phase 0's exit criteria before any Phase 1+ work starts, and update the project's knowledge documents so the fixed findings stop being reported as open.

## Exit criteria (from the phase plan)

1. **No request can hang the engine loop** — unknown family / unresolvable routing / expired deadline all resolve as typed errors.
2. **No client payload can execute code via deserialization** — all `torch.load` on request data is `weights_only=True` (or safetensors).
3. **No cross-request state leakage** — Chatterbox `conds`/`ref_dict` restored per request; missing conditioning fails loudly.
4. **No request/input mispairing** — explicit request passing in all handlers; mixed batches split; alignment asserted.

## Steps

1. **Run the full test suite** (`tests/unit`, `tests/integration`, `tests/e2e` — CPU) and record results. Per workspace rules, building/running is done manually — coordinate with the owner.
2. **Cross-review the diff** of all Phase 0 subtasks for reintroduced patterns: grep for `weights_only=False`, `initial_request` reads inside nested `def`s, `self.model.conds =` / `self.model.ref_dict =` outside the restore-wrapped handlers.
3. **Update `CONCERNS.md`** (`.planning/codebase/CONCERNS.md`): mark the addressed findings as resolved with a one-line reference to the fixing subtask; keep any findings that were only partially addressed, with a note.
4. **Update `memory-bank/`**: `progress.md` (Phase 0 complete), `activeContext.md` (current workstream pointer to this plan folder) — per workspace rules, check `.toon` files first if present.
5. **Update `CHANGELOG.md`** and, if the security fix is user-visible (it is — API behavior on bad payloads changes), `docs/` API notes.
6. **Tick all Phase 0 checkboxes** in [summary.md](summary.md) and note any deviations in the respective phase files.

## Checklist

- [ ] Full suite run recorded (pass/fail per suite)
- [ ] Anti-pattern grep sweep clean
- [ ] `CONCERNS.md` findings marked resolved
- [ ] `memory-bank/progress.md` + `activeContext.md` updated
- [ ] `CHANGELOG.md` / docs updated
- [ ] `summary.md` Phase 0 fully ticked; gate for Phases 1–5 declared open
