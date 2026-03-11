# State: Chatterbox TTS Integration

## Project Reference

**Core Value:** Integrate Chatterbox TTS model family (4 variants) into Harmony Speech Engine with voice cloning support

**Current Focus:** Initial roadmap created - awaiting approval to begin Phase 1

---

## Current Position

| Attribute | Value |
|-----------|-------|
| **Phase** | 0 - Not Started |
| **Plan** | None |
| **Status** | Ready to begin |
| **Progress** | 0/7 phases |

### Progress Bar

```
[════════════════════════════] 0%
```

---

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Requirements Mapped | 25 | 25 (100%) |
| Phases Defined | 7 | 7 |
| Success Criteria | 26 | 26 |
| Dependencies Identified | 6 | 6 (between phases) |

---

## Accumulated Context

### Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | 7-phase structure | Matches standard depth (5-8); natural delivery boundaries derived from requirements |
| D-02 | Phase ordering | Dependencies flow: Setup → Registration → Input → Execution → Routing → Config → Tests |
| D-03 | Phase 3 covers all input types | Single phase for TTS, embedding, VC inputs - coherent input preparation capability |
| D-04 | Phase 5 covers all routing | Single phase for all routing logic - enables end-to-end testing in later phases |

### Open Questions from Requirements

| ID | Question | Status |
|----|----------|--------|
| OQ-01 | Should multilingual model languages be exposed in model card voices list? | Open - depends on API design preference |
| OQ-02 | How to handle Turbo's ignored params - warn or silent ignore? | Open - need consistency with other model error handling |
| OQ-03 | Should we implement eager loading of Chatterbox models on startup, or lazy load? | Open - follow existing HSE pattern (lazy) |

### Dependencies Between Phases

| From Phase | To Phase | Dependency Type |
|------------|----------|-----------------|
| 1 - Dependencies & Setup | 2 - Model Registration | Required packages needed for model loading |
| 2 - Model Registration | 3 - Input Preparation | Models must be registered before inputs can prepare model-specific data |
| 3 - Input Preparation | 4 - Model Execution | Input preparation outputs feed into execution |
| 4 - Model Execution | 5 - Request Routing | Routing dispatches to execution methods |
| 5 - Request Routing | 6 - Configuration & Performance | Routing enables full pipeline testing |
| 6 - Configuration & Performance | 7 - Testing & Documentation | Full pipeline ready for testing |

---

## Session Continuity

### Last Session

- **Date:** 2026-03-12
- **Action:** Created ROADMAP.md with 7 phases
- **Outcome:** Roadmap draft ready for review

### Next Steps

1. Review and approve roadmap
2. Begin Phase 1: Dependencies & Setup
3. Execute `/gsd-plan-phase 1` to create implementation plan

---

## Notes

- **Project Type:** Feature Integration (not greenfield)
- **Complexity:** Medium-High (4 model variants, multilingual, voice cloning)
- **Risk Areas:** 
  - Voice cloning multi-step routing (RQ-ROUTE-01, RQ-ROUTE-04)
  - Conditionals serialization/deserialization (RQ-EXEC-02)
  - 23-language support validation (RQ-INPUT-02)
- **Config Setting:** `nyquist_validation: true` enabled

---

*Last updated: 2026-03-12*