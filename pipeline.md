# Dubbing pipeline

How a video becomes a dubbed (and optionally lip-synced) video in the studio. This is the reference for changing
the pipeline: **any change to a stage, its order, its models, its key constants, its files on disk, its run.json /
chunks.json fields or its endpoints must be reflected here in the same change** (see CLAUDE.md).

Code: [backend/](backend/) (FastAPI app, `uvicorn backend.main:app`, port 8000, started by `run_app.sh`, which pins the backend to the GPU with the most free memory at each (re)start via
`CUDA_VISIBLE_DEVICES` unless it is already set),
[lipsync/](lipsync/) (GPU subprocesses in the LatentSync venv), UI in [ui/index.html](ui/index.html).

## Overview

```
upload / YouTube fetch
  └─ POST /v2/runs ─────────────── _diarize_step (automatic)
       trim → extract audio → separate (vocals / crowd / bg) → diarize (Nemotron) + gender
       → faces + Light-ASD fixes → portraits → pause-split pieces            ⇒ stage "diarized"
  └─ POST /v2/runs/{id}/transcribe ─ _asr_step
       ASR per piece (Bodhan) → retry failed once → Gemma cut check → short joins
       → merge → clean-up: punctuation, misheard words (Gemini)   ⇒ stage "transcribed"
       └─ (if target_language) _translate_step
            batched translation (Bodhan) → translation check (review.py)     ⇒ stage "translated"
  └─ POST /v2/runs/{id}/dub ──────── _dub_step
       TTS per chunk (Bodhan) → cadence fit per chunk → mix + mux            ⇒ stage "dubbed"
  └─ POST /v2/runs/{id}/lipsync ──── _lipsync_step
       prepass (faces, ASD, ranges) → LatentSync worker + SyncNet gate → paste back → lipsynced.mp4
```

Stages (`studio.STAGES`): `diarized` → `transcribed` → `translated` → `dubbed`. `info["stage"]` is the furthest
reached; redoing an earlier stage keeps later results but lists them in `info["stale"]` (`translation`, `dub`,
`lipsync`) until they are run again (`_mark_stale` / `_unstale`).

## Orchestration ([backend/studio.py](backend/studio.py))

- Router `/v2`, wired up by [backend/main.py](backend/main.py) via `studio.configure(...)`, which passes in
  `_trim`, `extract_audio`, `separate` and the ffmpeg binary.
- Each step runs in a background thread (`_start`). The UI polls `GET /v2/runs/{id}`.
- In-memory `runs[run_id]` records are replaced, never mutated (`_set`), each with a `version`. `_claim` refuses to
  start a step while the run's status is in `RUNNING` (409).
- On disk each run is `storage/<32-hex run_id>/` with `run.json` (the run info) and `chunks.json` (the segments).
  `_load` reads them back (and `_upgrade`s old chunk formats); `_record` builds the full state the UI gets;
  `_segments` builds the UI rows (one per merged chunk).
- A step that raises sets `status="error"`, `error`, `failed_at`.
- Lip-sync and the faces step share one GPU slot app-wide (`_lipsync_slot = Semaphore(1)`); they run as
  subprocesses via `lipsync/run.sh` (`_lipsync_proc`, which also publishes `lipsync/progress.json` every 2 s).

## Stage 1 — ingest, separation, diarization (`_diarize_step`)

Triggered by `POST /v2/runs` (multipart: `file` **or** `fetch_id`, `language`, optional `range_start`/`range_end`).
Checks the Bodhan ASR key up front.

1. **Input.** Upload saved as `input.<ext>`; if a range is given it is saved as `original.<ext>` and
   **trimmed** (`main._trim`, ffmpeg re-encode, libx264 crf 18 + aac) to `input.mp4`. YouTube:
   [backend/youtube.py](backend/youtube.py) probes (`POST /v2/youtube/info`), downloads only the chosen part
   (≤720p, ≤60 min, into `storage/_youtube/<fetch_id>/`, `POST /v2/youtube`, poll `GET /v2/youtube/{id}`), then
   `youtube.take` moves it into the run folder.
2. **Extract audio** (`main.extract_audio`): `audio.wav`, stereo 44.1 kHz.
3. **Separate** (`main.separate`): Mel-Band RoFormer `melband-roformer-kim-vocals` → vocals; then (if
   `CROWD_SEPARATION`) `mel-roformer-crowd-aufr33-viperx` on the vocals pulls cheering/laughter out
   (`crowd.wav`), which then stays in the background. Writes `vocals.wav`, `bg_noise.wav` (= mix − vocals).
   Batched overlap-add (`_demix`, `SEPARATOR_BATCH_SIZE=4`), bf16 autocast, models warmed up at startup.
   Checkpoints in `models/`.
4. **Diarize** ([backend/diarization.py](backend/diarization.py) `diarize_vocals`): vocals → `vocals_16k.wav`
   (mono 16 kHz, used by every later step as "the original speech"); `nvidia/Nemotron-3-Diarization`; overlaps
   flattened, segments with < `MIN_VOICED` voiced frames dropped (stem bleed), short ones cleaned
   (`MIN_SEG`, `INTERJECTION`), speakers relabelled 1..n by first appearance.
   **Gender** ([backend/gender.py](backend/gender.py), audeering wav2vec2 age-gender, ≤30 s per speaker,
   `MIN_CONF=0.65` else `unknown`). Writes `diarization.json` (`turns`, `genders`, …).
5. **Faces / ASD** (`_faces_step` → `lipsync/run.sh faces`, [lipsync/faces.py](lipsync/faces.py)): shots, face
   tracks, people (ArcFace), Light-ASD speaking scores → `faces.json`, plus speaker fixes (split / merge / relabel
   by whose face clearly speaks). `diarization.apply_faces` records them in `diarization.json["asd"]` and, if
   `ASD_APPLY`, uses the fixed turns (old ones in `turns_raw`, genders redone). On failure the audio-only turns
   stay. `faces.json` is reused by the lip-sync prepass.
6. **Portraits** ([backend/portraits.py](backend/portraits.py)): a 128 px face crop per speaker clearly seen
   talking → `portraits/<speaker>.jpg` + `index.json` (served at `/v2/runs/{id}/speakers/{n}/face`).
7. **Pause-split chunking** ([backend/cuts.py](backend/cuts.py) `split_pieces`): same-speaker turns < 0.5 s
   apart joined; each turn cut at every silence ≥ `SPLIT_PAUSE_S=0.5`; pieces still longer than the ASR limit
   (`diarization.MAX_AUDIO=15 s` incl. `PAD=0.15`) cut by `diarization._split`; pieces < `MIN_PIECE_S` dropped.
   Each piece has `pause_after`, `forced_after`, `cut_start`/`cut_end`. Writes `pieces.json` and `chunks.json`.

→ `run.json`: `stage="diarized"`, `video`, `num_speakers`, `timings.separation/diarization`.

## Stage 2 — ASR, cut check, transcript clean-up (`_asr_step`)

`POST /v2/runs/{id}/transcribe` with optional `target_language`, `redo`, `language`.

1. **ASR** (`cuts.transcribe_pieces`, Bodhan `indic-transcribe`, key from env `BODHAN_ASR_KEY` or `bodhan.txt`):
   each piece's `[cut_start, cut_end]` of `vocals_16k.wav` → `chunks/<id>.wav` → one request. Junk transcripts
   (repetition loops, wrong script; `asr._junk_reason`) count as failures. 429s pause and resend (not a failure).
   Each segment is published as it arrives. Failed pieces get **one** more try `ASR_RETRY_DELAY_S=180` s after
   they failed. Raw transcripts saved to `chunks_asr_raw.json` (a full `redo` restarts from it).
2. **Cut check** (`cuts.check_cuts`, OpenRouter `google/gemma-4-26b-a4b-it`, reasoning off): every same-speaker
   cut judged keep / join (join if translating the sides separately would hurt), 40 pieces per request + 3
   context each side. Uses the target language (or `DEFAULT_TARGET="hi"`). Raw replies in `cut_check_raw/`.
3. **Short joins** (`cuts.short_joins`): a segment < `SHORT_CHARS=100` chars joins a same-speaker neighbour
   ≤ `SHORT_GAP_S=1.5` s away, with a comma; Gemma picks the side when both qualify (`short_check_raw/`).
4. **Merge** (`asr.merge_chunks(joins, commas)`): chunks renumbered 0..M-1, the original pieces kept in
   `sub_chunks`. The UI's segment rows are only updated here: during ASR and the cut checks it gets the progress
   counts (`asr_done`) but not the unmerged pieces' texts. On a re-run, `_merge_again` carries over the earlier merged chunks' translation/dub and marks
   changed ones `retranslate`/`refresh`.
   **Slow speech** (`asr.mark_slow`, at the end of every merge), from pauses, not word rate (which depends on the
   language): a merged chunk gets `pause_share` (silence between its pieces / span), `wps` (logged only) and
   `slow` = span ≥ `SLOW_MIN_SPAN_S=6` s, a pause between pieces ≥ `SLOW_PAUSE_S=1.0` s and `pause_share` ≥
   `SLOW_PAUSE_SHARE=0.2`. Each piece gets `wps`, `pause_after` and `slow` (its chunk is slow and the pause after
   it ≥ `SLOW_PAUSE_S`). A slow chunk changes the clean-up prompt, the check and the fit (below). Thresholds were
   set on English runs only.
5. **Clean-up** (`asr.punctuate_chunks`, OpenRouter `google/gemini-3.5-flash-lite`, no `reasoning` field sent
   (`{"enabled": false}` is refused), `PUNCT_MAX_TOKENS=16000`, JSON mode;
   `PUNCT_WINDOW=25` chunks/request + `PUNCT_CONTEXT=2` read-only neighbours each side; the run's `name` (video
   title) is sent as the topic). An editable slow chunk is also sent with `slow` and `parts` (each piece's text,
   `pause_after_s`, `slow`); the prompt asks for a stopping point only after a slow piece: a comma where a written
   one could stand (none after an article, preposition, auxiliary, conjunction or inside a noun phrase), or a
   sentence end between two complete clauses; elsewhere it punctuates as usual. One call does: misheard words (sound-alike words, misspelt/split terms, names,
   numbers, wrong script), punctuation and broken
   contractions (numbers and symbols kept as spoken words), and removal of hesitations, stutters and filler tics. No rephrasing or grammar fixes. `text_raw`
   keeps the text before. A reply is dropped (`_light_edit`, word diff ignoring case and punctuation) if it keeps
   under half the words of a chunk of more than 6, or adds more than `max(PUNCT_MIN_NEW=3, PUNCT_MAX_NEW=0.3 ×
   words)` new words. Chunks done get `punctuated` (a retry only sends the rest); `text_edited` chunks are skipped.
   An already translated chunk whose text changed gets `retranslate` + `refresh`. Failure is non-fatal. Stats in
   `info["punctuation"]` (`model`, `changed`, `dropped`, `failed`, `requests`, `attempts`, `slow_chunks`,
   `slow_sub_chunks`, or `error`). Raw in
   `punctuate_raw/`.
6. `_split_punctuated` gives each sub-chunk its share of the cleaned words.

→ `stage="transcribed"`, `asr_failed`, `cut_check` and `punctuation` stats. If a target was given, continues into Stage 3.

## Stage 3 — translation and translation check (`_translate_step`, `_review_step`)

`POST /v2/runs/{id}/translate` (`target_language`, `redo`) or chained from ASR. Same language as source = text
copied, no requests, no check.

1. **Translate** ([backend/translate.py](backend/translate.py) `translate_chunks`, Bodhan `indic-translate`,
   script `codemix`): chunks packed into batches (≤700 chars / ≤30 chunks, `[i]` line markers); a bad batch is
   split in half and retried; rate limit 15 requests / 61 s. Long texts split at sentence/clause marks.
   Retries only send untranslated or failed chunks. `map_sub_translations` slices each merged translation back
   onto its sub-chunks (sentence-aligned). Raw in `translation_raw/`.
2. **Check** ([backend/review.py](backend/review.py) `review_chunks`, OpenRouter `google/gemini-3.5-flash-lite`,
   25 per window + 2 context): flags/rewrites `too_long` (vs. slot at `TARGET_SPEED=1.4` and ~14 chars/s),
   `punctuation`, `broken_meaning`, `mistranslation`, `too_literal`, `untranslated`, `wrong_gender` (speaker
   gender is sent; `slow` is sent too, and the prompt keeps the extra stopping points of slow segments). Edits applied at once; `translation_original` keeps the pre-check text (undoable).
   User-edited segments are skipped. The system prompt can be overridden per run
   (`PUT /v2/runs/{id}/review-prompt`, `{source}`/`{target}` placeholders). `POST /v2/runs/{id}/review` retries
   or (`redo`) resets and redoes it. Failure leaves translations as they were (`info["review"].error`).
   Raw in `review_raw/`.

→ `stage="translated"`; if the check changed translations of a dubbed run, `dub`/`lipsync` go stale.

## Stage 4 — TTS and dub (`_dub_step`, `_fit_step`)

`POST /v2/runs/{id}/dub` with `voices` (`{"<speaker>": voice}`) and `style` (one of `tts.STYLES` or null).
Pending `retranslate` chunks are translated first.

1. **Voices** ([backend/tts.py](backend/tts.py)): speakers ordered by talk time (`by_talk_time`), each gets the
   best unused voice of their detected gender from `preferred(lang)` (`PREFERENCE`, `REST_ORDER`); unknown gender
   → the user picks. English targets use the Hindi pool (`FALLBACK_LANG`). Voice previews:
   [backend/samples.py](backend/samples.py) (`/tts-sample/{lang}/{voice}.wav`, Gemma-written sentence).
2. **TTS** (`synthesize_chunks`, Bodhan `indic-speak`, `instructions={"lang", "style"?}`): one clip per merged
   chunk → `tts/<id>.wav`. Text > `MAX_TTS_CHARS=200` is split at sentence/clause/word and rejoined with
   0.6/0.3/0.15 s pauses (edge silence trimmed). `reuse=True` keeps clips with the same text+voice+style
   (`tts_text`, `tts_voice`).
3. **Fit** ([backend/dubbing.py](backend/dubbing.py) `build_speed_dub`, no API calls, `FIT_DESCRIPTION`): each
   chunk alone — edge silence trimmed, gain matched to the original speaker (0.25–4×), sped up just enough to fit
   its slot (never slowed, ≤ `MAX_SPEED=1.65`; floor `SPEED_FLOOR=1.1` for all if any needs >
   `SPEED_FLOOR_FROM=1.35`), via ffmpeg `atempo`. **Cadence**: TTS pauses paired with the original's by speech
   progress (DP over anchors, `_cadence_fill`) and stretched so speech resumes where the speaker's does;
   unmatched TTS pauses shrunk when a speed-up is needed; up to 0.4 s of original onset kept. A matched non-boundary
   pause grows to at most `min(max(x, y) + 0.3, STRETCH_MAX=1.5, max(x + 0.3, 2x))`; pairing needs the TTS pause ≥
   `MATCH_MIN_RATIO=0.4` × the original's; tail pauses left unmatched take `UNMATCHED_STRETCH_WEIGHT=0.05` up to
   `UNMATCHED_STRETCH_MAX_S=0.30`. **Slow chunks** (`c["slow"]`) whose TTS already fits the slot (one that needs a
   speed-up is fitted with the normal rules): no speed floor; only original pauses ≥ `SLOW_ORIG_PAUSE_S=1.0` get
   the slow rules: they also pair with a TTS pause ≥ `MATCH_MIN_RATIO` × `SLOW_MATCH_REF_MAX_S=0.75` if within
   `SLOW_PAIR_TOLERANCE=0.08` in progress, and their matched pause grows to `min(max(x, y) + 0.3,
   SLOW_STRETCH_MAX=3.0)` (no 2x cap); unmatched tail pauses take `SLOW_UNMATCHED_STRETCH_WEIGHT=0.25` up to
   `SLOW_UNMATCHED_STRETCH_MAX_S=1.20`; all pauses together grow by at most the original's total pause time in the
   chunk. `chunk["dub"]` adds `slow`, `relaxed_pairs` (pairs only the slow rule allowed) and `max_pause_added`;
   `info["dub"]` adds `slow_fitted`, `relaxed_pairs`, `max_pause_added`. Clips →
   `dub/<id>.wav` at 24 kHz, placed at chunk start. `chunk["dub"]` records slot, speed, gain, matched pauses, …
4. **Mix** (`_mix_and_mux`): dub track → 44.1 kHz stereo + `bg_noise.wav` + original vocals in the gaps between
   segments (`GAP_GAIN`, guarded/faded) → look-ahead limiter (0.97) → `dubbed_vocals.wav`, `dubbed_audio.wav`,
   `dubbed.mp4` (dubbed audio default track, original as second track).

`POST /v2/runs/{id}/refit` redoes only the fit + mix from the stored clips. Re-dubbing clears any lip-sync.

→ `stage="dubbed"`, `info["dub"]` (summary + `speed_floor`), `voices`, `tts_style`, `tts_failed`.

## Stage 5 — lip-sync (`_lipsync_step`)

`POST /v2/runs/{id}/lipsync` (needs `dubbed.mp4`). Queued on the app-wide GPU slot; `lipsync/run.sh` picks a GPU
with no other compute process (the app's own backend, its parent, is allowed: lip-sync may share its GPU,
which risks OOM since lip-sync peaks at ~10–11.5 GB next to the backend's ~6 GB) and ≥ 8 GB free, caps RAM (`LIPSYNC_MEM_MAX`, default 12G) via systemd-run, uses
the LatentSync venv at `/data1/pratyaymishra_data/latentsync/venv`.

1. **Prepass** ([lipsync/prepass.py](lipsync/prepass.py)): reuses `faces.json` (else: PySceneDetect shots, SCRFD
   faces, tracks, ArcFace people, Light-ASD scores at 25 fps); assigns speakers to people; builds speaking ranges
   per speaker (their segments ∪ where their dub plays, padded/joined, cut to shots and tracked faces, ≤ 30 s).
   → `lipsync/plan.json`.
2. **Worker** ([lipsync/worker.py](lipsync/worker.py)): per range a fixed head crop + that speaker's dubbed voice
   (from `dub/*.wav`) → LatentSync 1.5 (10 steps, guidance 1.5, DeepCache) → `lipsync/clips/<id>.mp4`.
   **SyncNet gate** ([lipsync/syncscore.py](lipsync/syncscore.py)): kept only if confidence ≥ `KEEP_MIN=3.0` and
   ≥ `KEEP_GAIN=1.5` above the unsynced crop. Kept clips pasted back into `dubbed.mp4` (feathered, 4-frame fades)
   → `lipsynced.mp4`. `lipsync/clips.json`, `progress.json`.

→ `info["lipsync"]` (people, ranges, kept/rejected, times, peak RAM/VRAM).

## Edits and single-segment refresh

- `PUT /v2/runs/{id}/segments/{seg}/text` — own transcript; marks `text_edited`, and once translated
  `retranslate` + `refresh`.
- `PUT …/translation` — own translation; `translation_edited`; in a dubbed run `refresh`.
- `PUT …/speaker` — move a chunk to another speaker; in a dubbed run `refresh` + `moved_from`.
- `POST /v2/runs/{id}/speakers/{n}/gender` — manual gender in `diarization.json` (`manual: true`); re-suggests that
  speaker's voice.
- `POST …/segments/{seg}/refresh` (`_refresh_step`) — only that chunk: retranslate if needed; in a dubbed run TTS
  with the run's voices/style, `dubbing.refit_segments` with the dub's speed floor (rest of the dub untouched,
  track rebuilt from `dub/*.wav`); if lip-synced, prepass again and `worker --update t0,t1 --speakers …`.
  `info["refreshes"]` bumps (UI cache key).
- User-edited segments are never sent to the translation check.

## Run folder (`storage/<run_id>/`)

| File | Made by |
|---|---|
| `original.*`, `input.*` | upload / trim |
| `audio.wav`, `vocals.wav`, `crowd.wav`, `bg_noise.wav` | extract, separate |
| `vocals_16k.wav`, `diarization.json` | diarization (+ gender, ASD fixes) |
| `faces.json`, `faces.log`, `portraits/` | faces step, portraits |
| `pieces.json`, `chunks.json` | chunking; `chunks.json` is then updated by every step |
| `chunks/<id>.wav`, `chunks_asr_raw.json` | ASR |
| `cut_check_raw/`, `short_check_raw/`, `punctuate_raw/` | cut check, short joins, transcript clean-up |
| `translation_raw/`, `review_raw/` | translation, check |
| `tts/<id>.wav`, `dub/<id>.wav` | TTS, fit |
| `dubbed_vocals.wav`, `dubbed_audio.wav`, `dubbed.mp4` | mix |
| `lipsync/` (`plan.json`, `clips/`, `clips.json`, logs), `lipsynced.mp4` | lip-sync |
| `run.json` | run info: stage, stale, timings, language/target, voices, dub, lipsync, … |

Downloads: `GET /download/{run_id}/{input|vocals|bg_noise|dubbed_audio|dubbed_vocals|dubbed_video|lipsynced_video|diarization|chunks|pauses}`.

## External services and keys

| Service | Used for | Key |
|---|---|---|
| Bodhan `indic-transcribe` | ASR | `BODHAN_ASR_KEY` or `asr:` line in `bodhan.txt` |
| Bodhan `indic-translate` | translation | `BODHAN_TRANSLATE_KEY` or `translate:` line |
| Bodhan `indic-speak` | TTS | `BODHAN_TTS_KEY` or `tts:` line |
| OpenRouter (Gemma 4 26B, Gemini 3.5 flash-lite) | cut check, short joins, transcript clean-up, translation check, voice-sample texts | `OPENROUTER_API_KEY` or `openrouter:` line |
| Hugging Face | Nemotron diarization weights | `HF_TOKEN` |

Local models: `models/` (RoFormer vocals + crowd, wav2vec2 age-gender, …), LatentSync / Light-ASD under
`/data1/pratyaymishra_data/latentsync/`.
