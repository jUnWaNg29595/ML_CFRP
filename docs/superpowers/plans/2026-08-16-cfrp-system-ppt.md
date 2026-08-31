# CFRP 系统全流程科研汇报 PPT Implementation Plan

> **For agentic workers:** This plan is executed inline in the current session. It produces a standalone presentation and does not modify application behavior.

**Goal:** Create and verify a 32-page Chinese academic-method presentation that explains the CFRP platform end to end, with detailed coverage of molecular feature engineering and formulation-level high-throughput screening.

**Architecture:** Build the deck in an external scratch workspace using PptxGenJS and `PPT-master`'s slide conventions. Use reusable layout helpers for headers, footers, section dividers, process diagrams, tables, and callouts; keep all system claims grounded in the current repository documentation and code structure. Export only the final `.pptx` into the repository `outputs` directory.

**Tech Stack:** Node.js, PptxGenJS, `PPT-master` lint/audit tools, LibreOffice or the available slide renderer, Microsoft YaHei, Arial.

## Global Constraints

- Do not modify `app.py`, `core/`, tests, model artifacts, caches, backups, or any user-owned uncommitted files.
- Do not include active learning, fabricated experimental metrics, fabricated screening hit rates, or unsupported performance claims.
- Use 16:9 canvas, `Microsoft YaHei` for Chinese, `Arial` for English/numbers, and the approved deep-blue/teal/orange palette.
- Keep generated source, previews, QA logs, and intermediate assets outside the repository.
- Run `node scripts/lint.js` before compilation and inspect rendered slides before delivery.

## Files

- Create: `C:\Users\wangj\AppData\Local\Temp\cfrp-system-ppt\slides\deck.js` — complete deck source.
- Create: `C:\Users\wangj\AppData\Local\Temp\cfrp-system-ppt\source-notes.txt` — verified content notes and source paths.
- Create: `C:\Users\wangj\AppData\Local\Temp\cfrp-system-ppt\qa\` — render and audit outputs.
- Create: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\outputs\CFRP系统全流程-科研方法汇报-2026-08-16.pptx` — final deliverable.
- Do not modify: existing application files or the user's uncommitted changes.

### Task 1: Prepare the external presentation workspace

- [ ] Create the temporary workspace directories and verify the final output directory.
- [ ] Install `PPT-master/assets` dependencies only if its `node_modules` directory is absent.
- [ ] Confirm the installed `pptxgenjs` package can be required by Node.js.

### Task 2: Capture verified system content

- [ ] Extract the current system workflow facts from `README.md`, `CHANGELOG.md`, `docs/process-pls-workflow.md`, `app.py` page definitions, and relevant `core/` modules.
- [ ] Record the exact terms used for SMILES/BigSMILES, multi-component roles, workflow replay, xTB, PLS, feature contracts, PubChem, resin/hardener pools, staged generation, and result ranking.
- [ ] Mark unsupported values as conceptual; do not present counts or model metrics unless they are explicitly present in the source materials.

### Task 3: Implement the narrative and reusable slide primitives

- [ ] Define the theme, font constants, slide metadata, safe-area constants, and page numbering.
- [ ] Implement reusable helpers for chapter tags, titles, subtitles, footers, process nodes, branch diagrams, comparison rows, code-like feature labels, and callout markers.
- [ ] Implement the 32-page sequence in five narrative blocks: background, data/feature overview, molecular feature engineering, process/model workflow, and high-throughput screening/closure.
- [ ] Use editable PowerPoint text and shapes for simple diagrams; use compact SVG assets only where a molecular or process visual is clearer than native shapes.
- [ ] Keep each slide to one main conclusion and make the two focus areas visually denser than supporting sections without reducing legibility.

### Task 4: Compile and run static quality checks

- [ ] Run `node C:\Users\wangj\.codex\skills\PPT-master\scripts\lint.js` against the generated source or the skill workspace copy.
- [ ] Compile the deck with PptxGenJS and verify that the output exists and is non-empty.
- [ ] Run the PPTX metadata audit and text extraction; check for missing page titles, replacement characters, unsupported claims, and absent key terms.

### Task 5: Render, inspect, and revise

- [ ] Render every slide to PNG using the available `PPT-master` or LibreOffice-compatible renderer.
- [ ] Create a contact sheet for deck-level flow review and inspect full-size pages for clipping, overlaps, text wrapping, contrast, and diagram alignment.
- [ ] Fix every unintended layout issue found by rendering or audit, rerun lint/compile, and preserve the final QA outputs in the external workspace.

### Task 6: Deliver the final deck

- [ ] Copy only the verified final `.pptx` to the repository `outputs` directory.
- [ ] Report the exact absolute file path, page count, and validation commands/results.
- [ ] Leave all unrelated user changes untouched and do not commit or push unless explicitly requested.

