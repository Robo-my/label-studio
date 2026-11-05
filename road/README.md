# Road damage VLM: sample SFT and GRPO data

All records are made-up examples that show the layout. The crop files they name are placeholders.

| File | What it holds |
|---|---|
| `sft_sample.jsonl` | 12 SFT records: 7 defects and 5 `no_defect`. Each has the question and the full answer (label and caption). |
| `grpo_sample.jsonl` | 8 GRPO records. Each has the question and the correct label (`solution`) only, sorted by uncertainty. |
| `reward.py` | The GRPO reward, and the uncertainty score that picks GRPO records. Run `python reward.py` to self-test. |
| `make_crop.py` | Makes the model input: the box plus 30% context, a red outline, long side 512 px. |

## Labels

`pothole`, `crack`, `raveling`, `no_defect`. Manhole and stagnant water boxes become `no_defect`, and the caption names them.

## Record layout

- `messages`: the question, plus the answer in SFT records only.
- `images`: one crop path. Make the crops with `make_crop.py`.
- `solution`: GRPO records only. The correct label that `reward.py` compares with.
- `meta`: where the record came from. Trainers do not read it; remove it if your trainer rejects unknown keys.

Field names follow the common `messages` + `images` layout. Rename keys if your trainer expects other names.

## Caption rules

The SFT captions and the GRPO reward use the same rules:

- 15 words or fewer.
- Names its own label, or starts with "No defect".
- No severity or size words, and no numbers.
- Never names a different defect.

## Picking GRPO records

1. Keep checked records out of SFT.
2. Let the SFT model answer each one 8 times.
3. Score the 8 answers with `reward.py`.
4. Keep records with 1 to 7 correct answers, and rank them by `uncertainty()`.

Records with 0 or 8 correct answers have an uncertainty of 0. They teach GRPO nothing.
