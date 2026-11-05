// End-to-end test: a real browser works through 5 tasks in Label Studio with road-captions.
// Start the stack first (see README "Test the screen"), then:
//   node e2e/run_e2e.js [http://localhost:8080] [screenshot folder]
// Needs Playwright with Chromium. Uses the admin login from e2e/e2e.env.
const { chromium, request } = require("playwright");
const fs = require("fs");
const path = require("path");

const BASE = process.argv[2] || "http://localhost:8080";
const SHOTS = process.argv[3] || path.join(__dirname, "out");
const HERE = __dirname;
const env = Object.fromEntries(fs.readFileSync(path.join(HERE, "e2e.env"), "utf8").split("\n")
  .filter((l) => l.includes("=")).map((l) => [l.split("=")[0], l.slice(l.indexOf("=") + 1)]));
const results = [];
function check(name, ok, detail) {
  results.push({ name, ok: !!ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  -- " + detail : ""}`);
}

async function main() {
  fs.mkdirSync(SHOTS, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({ baseURL: BASE, viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2 });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on("console", (m) => { if (m.type() === "error" || m.text().includes("road-captions")) consoleErrors.push(m.text()); });
  let sentenceRequests = 0;
  page.on("request", (r) => { if (r.url().includes("/road-api/sentence")) sentenceRequests++; });

  // ---- log in and set up the project through the API, with the session cookie ----
  await page.goto("/user/login/");
  await page.fill('input[name="email"]', env.LS_ADMIN_EMAIL);
  await page.fill('input[name="password"]', env.LS_ADMIN_PASSWORD);
  await Promise.all([page.waitForURL((u) => !u.pathname.startsWith("/user/login")),
    page.click('button[type="submit"], input[type="submit"]')]);
  const csrf = (await context.cookies()).find((c) => c.name === "csrftoken").value;
  const api = async (method, url, data) => {
    const r = await page.request.fetch(url, { method, data, headers: { "X-CSRFToken": csrf, Referer: BASE + "/" } });
    if (!r.ok()) throw new Error(`${method} ${url} -> ${r.status()} ${await r.text()}`);
    return r.json();
  };
  const config = fs.readFileSync(path.join(HERE, "..", "config.xml"), "utf8");
  const project = await api("POST", "/api/projects", { title: "Road review e2e " + Date.now(), label_config: config });
  await api("POST", "/api/storages/localfiles", { project: project.id, title: "e2e images", path: "/label-studio/files/e2e", use_blob_urls: false });
  const tasks = JSON.parse(fs.readFileSync(path.join(HERE, "tasks.json"), "utf8"));
  await api("POST", `/api/projects/${project.id}/import`, tasks);
  check("project created with config.xml and 5 tasks imported", true, `project ${project.id}`);

  // ---- the service refuses requests without a login or without the header ----
  const anon = await request.newContext({ baseURL: BASE });
  const r401 = await anon.post("/road-api/sentence", { data: { task_id: 1, decision: "Correct" }, headers: { "X-Road-Captions": "1" } });
  check("no login -> 401", r401.status() === 401, `got ${r401.status()}`);
  const r403 = await page.request.post("/road-api/sentence", { data: { task_id: 1, decision: "Correct" } });
  check("logged in, no X-Road-Captions header -> 403", r403.status() === 403, `got ${r403.status()}`);

  // ---- label like an intern ----
  await page.goto(`/projects/${project.id}/data?labeling=1`);
  await page.waitForFunction(() => window.Htx && window.Htx.task && window.Htx.annotationStore.selected
    && window.Htx.annotationStore.selected.names.get("decision"), null, { timeout: 60000 });
  check("road-captions.js loaded in the Label Studio page", await page.evaluate(() => !!window.__roadCaptions));

  const state = () => page.evaluate(() => {
    const s = window.Htx, a = s.annotationStore.selected, g = (n) => a.names.get(n);
    const panel = document.querySelector("#road-sentence .rc-panel");
    return {
      rid: s.task.dataObj.record_id, taskId: s.task.id,
      decision: g("decision").selectedValues()[0] || null,
      lookalike: (g("lookalike") && g("lookalike").selectedValues()[0]) || null,
      words: g("words").regions.map((r) => r._value), caption: g("caption").regions.map((r) => r._value)[0] || "",
      source: g("caption_source").regions.map((r) => r._value)[0] || "",
      panel: panel ? panel.innerText.replace(/\s+/g, " ").trim() : "", panelClass: panel ? panel.className : "",
    };
  });
  const waitPanel = (cls, text) => page.waitForFunction(([c, t]) => {
    const p = document.querySelector("#road-sentence .rc-panel");
    return p && p.classList.contains(c) && (!t || p.innerText.includes(t));
  }, [cls, text || ""], { timeout: 15000 });
  const key = async (k) => { await page.locator("body").press(k); };
  const submit = async () => {
    const before = (await state()).taskId;
    await page.keyboard.press("Control+Enter");
    await page.waitForFunction((id) => window.Htx && window.Htx.task && window.Htx.task.id !== id
      && window.Htx.annotationStore.selected, before, { timeout: 15000 }).catch(() => null);
    return before;
  };
  const typeWords = async (text) => {
    const box = page.getByPlaceholder("Sentence wrong?", { exact: false });
    await box.click();
    await box.pressSequentially(text, { delay: 15 });
    await box.press("Enter");
  };
  const submitted = {};

  for (let i = 0; i < 5; i++) {
    await page.waitForTimeout(300);
    const s = await state();
    if (s.rid === "t1_pothole") {
      const t0 = Date.now();
      await key("1");
      await waitPanel("rc-ok", "Pothole");
      let st = await state();
      check("t1 key 1: model sentence shown and saved", st.caption === "Pothole with broken edges and water inside." && st.source === "bedrock",
        `${Date.now() - t0} ms, caption=${JSON.stringify(st.caption)}`);
      await page.screenshot({ path: path.join(SHOTS, "1-key-correct.png") });
      await typeWords("water inside, broken edges");
      await waitPanel("rc-ok", "water inside");
      st = await state();
      check("t1 typing words fired no hotkeys", st.decision === "Correct" && st.lookalike === null, JSON.stringify([st.decision, st.lookalike]));
      check("t1 words -> live rewrite keeps the words", st.caption === "Pothole with water inside, broken edges." && st.words[0] === "water inside, broken edges",
        JSON.stringify(st.caption));
      await page.screenshot({ path: path.join(SHOTS, "2-rewrite.png") });
    } else if (s.rid === "t2_manhole") {
      await key("1");
      await waitPanel("rc-ok", "manhole");
      const st = await state();
      check("t2 key 1 on a manhole: fixed look-alike sentence", st.caption === "No defect: manhole cover on intact asphalt." && st.source === "lookalike", st.caption);
    } else if (s.rid === "t3_crack") {
      await key("3");
      await waitPanel("rc-ok", "Crack");
      await typeWords("pothole edge");
      await waitPanel("rc-error", "pothole");
      let st = await state();
      check("t3 words naming another defect: refused, sentence cleared", st.caption === "" && st.panel.includes("another defect"), st.panel);
      await page.screenshot({ path: path.join(SHOTS, "3-refused.png") });
      await page.keyboard.press("Control+Enter");
      const modal = page.locator(".ant-modal-confirm-warning", { hasText: "The sentence is not ready yet" });
      await modal.waitFor({ timeout: 5000 }).catch(() => null);
      await page.waitForTimeout(600); // let the dialog finish its opening animation
      st = await state();
      check("t3 submit blocked while no sentence", (await modal.isVisible()) && st.rid === "t3_crack", `modal=${await modal.isVisible()}`);
      await page.screenshot({ path: path.join(SHOTS, "4-submit-blocked.png") });
      await page.getByRole("button", { name: "OK" }).click().catch(() => page.keyboard.press("Escape"));
      await page.waitForTimeout(300);
      await key("7");
      await waitPanel("rc-idle", "No sentence is needed");
      check("t3 key 7 (unsure): no sentence needed", (await state()).caption === "");
    } else if (s.rid === "t4_raveling") {
      await key("5");
      await waitPanel("rc-idle", "Press q-u");
      await key("u");
      await waitPanel("rc-ok", "No defect");
      const st = await state();
      check("t4 keys 5 then u (other): model writes a no-defect sentence", st.caption === "No defect: oil stain on smooth asphalt." && st.lookalike === "other", st.caption);
    } else if (s.rid === "t5_badbox") {
      await key("6");
      await waitPanel("rc-idle", "No sentence is needed");
      check("t5 key 6 (bad box): no sentence needed", (await state()).caption === "");
    } else {
      check("unexpected task", false, JSON.stringify(s));
      break;
    }
    submitted[s.rid] = await submit();
  }
  check("all 5 tasks submitted", Object.keys(submitted).length === 5, Object.keys(submitted).join(", "));

  // ---- reopen a submitted task: the saved sentence must stay, and no new sentence is asked for ----
  await page.goto(`/projects/${project.id}/data?task=${submitted.t1_pothole}`);
  await page.waitForFunction(() => window.Htx && window.Htx.annotationStore.selected
    && window.Htx.annotationStore.selected.names.get("caption"), null, { timeout: 60000 });
  sentenceRequests = 0;
  await page.waitForTimeout(2500);
  const reopened = await state();
  check("reopened t1 keeps its saved sentence", reopened.caption === "Pothole with water inside, broken edges.", JSON.stringify(reopened.caption));
  check("reopening a submitted task sends no sentence request", sentenceRequests === 0, `${sentenceRequests} requests`);
  await page.screenshot({ path: path.join(SHOTS, "5-reopened.png") });

  // ---- export for the converter ----
  const exported = await api("GET", `/api/projects/${project.id}/export?exportType=JSON&download_all_tasks=true`);
  fs.writeFileSync(path.join(SHOTS, "export.json"), JSON.stringify(exported, null, 1));
  check("export has 5 annotated tasks", exported.filter((t) => t.annotations && t.annotations.length).length === 5);
  const rcErrors = consoleErrors.filter((e) => e.includes("road-captions"));
  check("no road-captions errors in the browser console", rcErrors.length === 0, rcErrors.slice(0, 3).join(" | "));
  await browser.close();
  const failed = results.filter((r) => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(2); });
