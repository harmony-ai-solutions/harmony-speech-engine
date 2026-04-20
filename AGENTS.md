# AGENTS.md

This project uses Roo Code with custom rules stored in `.roo/rules/`. Read these files for initial context:

- **[workspace-rules.md](.roo/rules/workspace-rules.md)** - CLI usage, workspace navigation, memory bank and codebase mapping conventions
- **[memory-bank.md](.roo/rules/memory-bank.md)** - How to use the Memory Bank system for project context
- **[implementation-plans.md](.roo/rules/implementation-plans.md)** - Creating and managing implementation plans (manual plans only, not GSD)
- **[gsd-validation-notes.md](.roo/rules/gsd-validation-notes.md)** - Notes on GSD frontmatter validation quirks

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **harmony-speech-engine** (4750 symbols, 9468 relationships, 230 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/harmony-speech-engine/context` | Codebase overview, check index freshness |
| `gitnexus://repo/harmony-speech-engine/clusters` | All functional areas |
| `gitnexus://repo/harmony-speech-engine/processes` | All execution flows |
| `gitnexus://repo/harmony-speech-engine/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
