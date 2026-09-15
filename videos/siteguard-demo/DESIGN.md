# DESIGN.md — SiteGuard Agent demo video

## Aesthetic
Dark technical blueprint. Deep navy `#0a0e1a`, fine cyan grid, blueprint-line
accents. Type-led scenes with generous whitespace; footage scenes full-bleed
with a dark gradient scrim. Motion-graphics narration explaining the pipeline.

## Palette
- bg `#0a0e1a` · panel `#0f1626` · grid line `rgba(34,211,238,.055)`
- cyan `#22d3ee` (primary accent / info) · green `#4ade80` (safe / approved)
- amber `#fbbf24` (pending / review) · red `#f87171` (violation / escalate)
- orange `#f97316` (secondary accent) · text `#e8eef7` · dim `#93a1b8`

## Typography
- Headings: system-ui, weight 800, tight. Gradient white→ice-blue on hero titles.
- Labels/tags: ui-monospace, 26px, letter-spacing .35em, uppercase, cyan.
- Trace/code content: ui-monospace.
- No webfonts (deterministic offline render).

## Motion rules (from hyperframes-animation)
- kinetic-type-beats for problem / responsible-use / credits: statements arrive
  as timed beats synced to narration, never dumped at t=0.
- grid-card-assemble for trace + metrics: cards cascade in timeline order.
- Footage scenes: let real footage play; only captions + risk meter animate.
- Slow Ken Burns (28s, 1.0→1.12) over the architecture diagram.
- Scene transitions: 0.8s fade to black at each scene tail.

## Audio
- Narration: `tts` CLI, voice `avocado_v2:arista` (Lucid, Professional).
- 7 per-scene MP3s under `assets/audio/`, timed to scene starts +0.6s.
- No music (deliberate silence under narration; no royalty-safe bed resolved).
- Demo footage is muted (silent synthetic source).

## Determinism
- No Date.now / Math.random / network fetches in composition scripts.
- GSAP timelines paused, registered on `window.__timelines` per composition.
