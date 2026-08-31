# CFRP PPT Visual Refresh Implementation Plan

> **For agentic workers:** Use inline execution in the current session. This plan changes only the external PPT source and the final `.pptx`; do not modify `app.py`, `core/`, models, caches, or user data.

**Goal:** Replace repetitive card grids with scientifically meaningful visual diagrams while preserving the existing 32-page narrative and factual claims.

**Architecture:** Keep the existing PptxGenJS deck and helper functions, add reusable visual primitives for tracks, molecule nodes, funnels, queue lanes, matrix heat blocks, and circular closure diagrams, then replace the highest-impact slide builders in place. Recompile to the existing output filename and run the existing PPT-master QA tools.

**Tech Stack:** Node.js, PptxGenJS, PPT-master lint/audit scripts, existing temporary render workspace.

## Global Constraints

- Preserve 32 slides, 16:9, Microsoft YaHei/Arial, Chinese copy, and current factual scope.
- Do not add experimental metrics, screening hit rates, or unsupported numerical claims.
- Use deep blue for model/contracts, teal for data/features, yellow for decisions/warnings, and light gray only as a quiet background.
- Prefer arrows, tracks, branches, nodes, funnels, and rings over equal-width bordered cards.
- Keep all generated source and QA intermediates outside the repository except the final PPT and this documentation.

### Task 1: Add reusable scientific diagram primitives

**Files:**
- Modify: `C:\Users\wangj\AppData\Local\Temp\cfrp-system-ppt\slides\deck.js`

- [ ] Add primitives for a labeled track, molecule/network cluster, source funnel, queue lane, matrix heat block, and closure ring using existing `pptx.ShapeType` helpers.
- [ ] Keep primitive APIs synchronous and use only existing palette tokens.
- [ ] Compile the unchanged slide builders to confirm primitives introduce no syntax or dependency errors.

### Task 2: Redraw molecular workflow and model-process slides

**Files:**
- Modify: `C:\Users\wangj\AppData\Local\Temp\cfrp-system-ppt\slides\deck.js`

- [ ] Redraw slides 5, 10–18, and 20 with distinct visual grammars: end-to-end track, SMILES/BigSMILES split with molecule glyphs, role swimlanes, workflow timeline, method matrix, xTB pipeline, contract intersection, failure branch, and PLS track.
- [ ] Preserve all current labels and avoid implying unsupported algorithmic guarantees.
- [ ] Compile and inspect extracted slide text for lost or malformed content.

### Task 3: Redraw high-throughput screening and closure slides

**Files:**
- Modify: `C:\Users\wangj\AppData\Local\Temp\cfrp-system-ppt\slides\deck.js`

- [ ] Redraw slides 24–31 with a candidate funnel, source convergence, filter stack, design-space axes, batch queue, unique-structure reuse, weighted scoring lanes, and traceability ring.
- [ ] Keep the user-facing concepts of separate resin/hardener pools, PubChem limits, staged generation, pause/resume, feature reuse, and multi-objective ranking.
- [ ] Ensure the slide thumbnails visibly alternate between at least four layout families.

### Task 4: Compile and verify the final deck

**Files:**
- Create: `C:\Users\wangj\Desktop\CFRP系统\CFRP系统\outputs\CFRP系统全流程-科研方法汇报-视觉升级-2026-08-17.pptx`

- [ ] Run PPT-master lint against the temporary source copy.
- [ ] Compile with Node.js and confirm the output is non-empty.
- [ ] Run PPTX metadata audit and text extraction; verify 32 slides, 16:9, no replacement characters, and required focus terms.
- [ ] Render or inspect the available 32-slide QA image set and check title overlap, clipping, contrast, and diagram alignment.
