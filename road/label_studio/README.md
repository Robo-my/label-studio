# Label Studio review kit: live sentences

Checked:
- The full stack runs in Docker: stock Label Studio 1.23.2 on PostgreSQL, the `road-captions` container and nginx.
- A browser test (`e2e/run_e2e.js`) works through 5 tasks like an intern: 18 of 18 checks pass. It uses a fake model, not Bedrock.
- The export of that test goes through `ls_export_to_jsonl.py`: 3 verified, 1 to QA, 1 dropped, as expected.
- 43 unit tests for the service pass.

Not checked:
- A real Bedrock call. The model ID, latency and cost are not measured.
- Images from S3. The service follows Label Studio's presigned links. A unit test with a mocked redirect checks this; no real bucket was used.
- The `pip install` line in the Dockerfile. In this test, the image was built from wheels downloaded in advance; every other line is the same.

## How it works

```
intern's browser ── nginx ──┬── /road-api/*  → road-captions   (our features: sentence, checks, log)
                            └── everything else → label-studio (stock image, no code change)
```

| Container | What it does |
|---|---|
| `nginx` | One address for the interns. Sends `/road-api/*` to road-captions and everything else to Label Studio. Adds one script tag (`road-captions.js`) to Label Studio pages. |
| `label-studio` | Stock image `heartexlabs/label-studio:1.23.2`. Not changed. |
| `road-captions` | Writes the sentence live with a Bedrock model, checks it, and logs every request. Label Studio decides who may ask: the service sends the intern's login cookie to Label Studio for each request. |
| `db` | PostgreSQL for Label Studio. |

Only nginx opens a port. The other containers are reachable only inside the Docker network.

About the fork: our features need no change to Label Studio code. The browser module reads the editor through `window.Htx`. Label Studio sets that value itself, and its own test helpers read it (`web/libs/frontend-test/src/helpers/LSF/LabelStudio.ts`). You can keep your fork for later changes. If you change Label Studio code in the fork, build that image and put its name in `docker-compose.yml`.

## What the intern sees

| Key | Sentence |
|---|---|
| `1` correct | Sentence for the Detector's label. For manhole or stagnant water: the fixed "No defect: ..." sentence. |
| `2` / `3` / `4` | Sentence for pothole / crack / raveling, written live |
| `5`, then `q`–`y` | Fixed "No defect: ..." sentence for that look-alike |
| `5`, then `u` (other) | The model writes "No defect: ..." from the crop. Without the crop: the intern types the words. |
| `6` bad box, `7` unsure | No sentence |

The sentence appears in a box under the keys, with its source: model, template, look-alike, or the intern's own words.

When the sentence is bad, the intern types 2–6 words in the text box and presses Enter. Example: `water inside, broken edges`. A new sentence appears in about 1–3 seconds (not measured with Bedrock). It keeps every word the intern typed. Severity and size words and numbers are dropped on purpose.

Every sentence passes the caption rules in `../reward.py` before it is shown. These are the same rules as the GRPO reward. If no valid sentence is possible, the box turns red and gives the reason, for example "Your words name another defect (pothole)". Submit stays blocked until a sentence is ready, because the hidden `caption` field is required. To skip the sentence, the intern presses `7` (unsure).

Speed: 0.6 seconds after a task opens, the module asks for the Detector's sentence in advance. When the intern presses `1`, the sentence is usually ready.

A submitted task that is opened again keeps its saved sentence. The module does not ask for a new sentence until the intern presses a key.

## Files

| File | What it does |
|---|---|
| `docker-compose.yml`, `.env.example` | The four containers and their settings |
| `nginx/default.conf` | Routes and the script tag |
| `config.xml` | The intern screen. Paste it into the project's Labeling Interface. |
| `road_captions/app.py` | The service: `POST /road-api/sentence`, `GET /road-api/health` |
| `road_captions/sentences.py` | Labels, fixed sentences, prompts and checks. The converter uses the same file. |
| `road_captions/writer.py` | One Bedrock Converse call. Works with any Bedrock model that reads text and images. |
| `road_captions/ls_client.py` | Calls to Label Studio with the intern's login: who is this, may this person open this task, get the crop |
| `road_captions/static/road-captions.js` | The browser module |
| `prepare_tasks.py` | Detector boxes → Label Studio tasks. No model calls. |
| `ls_export_to_jsonl.py` | Exports → `verified.jsonl`, `sft.jsonl`, `grpo_candidates.jsonl`, `qa.jsonl`, `dropped.jsonl` and a report of golden accuracy and overlap agreement |
| `e2e/` | The browser test with a fake model |

## Setup

1. Start a small EC2 instance in `ap-southeast-5`, so the images stay in Malaysia. Give it an instance role that allows `bedrock:InvokeModel` on your model or inference profile. For a cross-region profile, the Bedrock docs list the extra model permissions. Do not put AWS keys in `.env`.
2. Let containers use the instance role. Set the metadata hop limit to 2:

   ```bash
   aws ec2 modify-instance-metadata-options --instance-id YOUR_INSTANCE_ID --http-put-response-hop-limit 2 --http-endpoint enabled
   ```

3. Copy `.env.example` to `.env` and fill it in. Set `BEDROCK_MODEL_ID` to the model ID or inference profile ID from your Bedrock console. Read the data residency section before you set `SEND_IMAGE`.
4. Start the stack from this folder:

   ```bash
   docker compose up -d --build
   ```

5. Open `PUBLIC_URL` and log in with `LS_ADMIN_EMAIL`. Sign-up without an invite link is off. Invite the interns from the Organization page.
6. Create one shared project for all reviewers. Paste `config.xml` into the Labeling Interface settings.
7. Connect the S3 bucket as source storage, with presigned URLs on. Do not sync it; the tasks come from step 8.
8. Write the tasks and import `tasks.json`. Include about 5% golden tasks from the golden train split, never from the Test split:

   ```bash
   python prepare_tasks.py --boxes boxes.jsonl --out tasks.json
   ```

9. Turn on 10% overlap and random order through the API. The free settings page does not show these fields. `LS_AUTH` is your authorization header, for example `Authorization: Bearer <access token>`:

   ```bash
   curl -X PATCH "$LS_URL/api/projects/$PROJECT_ID/" -H "$LS_AUTH" -H "Content-Type: application/json" -d '{"maximum_annotations": 2, "overlap_cohort_percentage": 10, "sampling": "Uniform sampling"}'
   ```

10. Keep golden answers only in `golden_answers.json` (`{"record_id": "label"}`). Never put them in task data, because every user can see task data.
11. Before the interns start, open 3 tasks yourself. Press `1`, type words and press Enter, and read the sentences.
12. Every evening, export the project as JSON and run:

    ```bash
    python ls_export_to_jsonl.py --exports exports/*.json --golden golden_answers.json --out out
    ```

13. The QA leads work through `out/qa.jsonl` and spot-check about 10% of `out/verified.jsonl`.

The request log shows every sentence request: who asked, which task, the key, the words, the sentence, its source, any model error, and the time in ms:

```bash
docker compose exec road-captions tail -f /var/log/road-captions/requests.jsonl
```

## Data residency

| Setting | What leaves Malaysia |
|---|---|
| `BEDROCK_MODEL_ID` empty | Nothing. All sentences are templates. |
| `SEND_IMAGE=false` (default) | Only the label and the intern's typed words, when the intern asks for a rewrite. Keys 1–4 give template sentences. |
| `SEND_IMAGE=true` | The crop of each box, for the model sentence. A cross-region inference profile can process it outside Malaysia. |

## Cost

Calls per box: one in advance for the Detector's label, one for each label change, and one for each rewrite (text only). The call in advance is wasted when the intern presses `5`, `6` or `7`.

Not verified: about 700 input and 100 output tokens per call with the crop, more if the model thinks first. At Anthropic list prices, Claude Haiku 5.5 costs about USD 2–3 per 20,000 boxes, and Claude Sonnet 5.5 about USD 48. Bedrock prices can differ: check the Bedrock pricing page.

## Test the screen

The test uses local images and a fake model (`e2e/fake_app.py`). It makes no Bedrock calls. Run it from this folder:

```bash
python3 e2e/make_files.py
```

```bash
docker compose --env-file e2e/e2e.env -f docker-compose.yml -f e2e/docker-compose.e2e.yml up -d --build
```

```bash
node e2e/run_e2e.js http://localhost:8080
```

Remove the test stack and its data:

```bash
docker compose --env-file e2e/e2e.env -f docker-compose.yml -f e2e/docker-compose.e2e.yml down -v
```

Unit tests for the service:

```bash
python -m pytest road_captions/tests
```

## Known limits

- The browser module uses editor internals: `window.Htx`, TextArea entries and `deleteArea`. It was tested on Label Studio 1.23.2. Pin that version, and run the browser test again before you upgrade.
- Label Studio 1.23.2 writes a console error when a text field loses its last entry: `AreaMixin.setValue` touches the deleted area. The module avoids that path.
- road-captions runs one worker process. Its cache and rate limits live in memory. That is enough for 13 interns.
- When you change the caption rules in `reward.py`, build the road-captions image again. Review and training then use the same rules.

## What open-source Label Studio covers

Checked in the upstream code: version 1.24.0.dev0, commit `0b31d3b`, Apache-2.0. The screen was tested on 1.23.2.

| Need | In open-source Label Studio | How this kit handles it |
|---|---|---|
| Task distribution | Built in: a task is locked while open, and nobody gets a task twice | One shared project |
| 10% overlap | Built in, API only: `maximum_annotations` and `overlap_cohort_percentage` | One `PATCH` call (step 9) |
| Hidden golden checks and accuracy per intern | Not built in: the scoring hooks are Enterprise-only | Golden tasks mixed in; the converter scores them |
| Agreement report | Not built in | The converter reports overlap agreement |
| Reviewer workflow | Not built in | The 2 QA leads work through `out/qa.jsonl` |
| Live sentence at the key press | No plugin hook | The road-captions container, plus one script tag that nginx adds |
