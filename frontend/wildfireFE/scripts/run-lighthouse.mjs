import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { launch } from "chrome-launcher";
import lighthouse from "lighthouse";

const [, , targetUrl, rawLabel = "audit"] = process.argv;

if (!targetUrl) {
  console.error("Usage: node scripts/run-lighthouse.mjs <url> [label]");
  process.exit(1);
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const reportsDir = path.join(projectRoot, "lighthouse-reports");
const label = rawLabel
  .trim()
  .toLowerCase()
  .replace(/[^a-z0-9-_]+/g, "-")
  .replace(/^-+|-+$/g, "") || "audit";
const reportStem = path.join(reportsDir, `${label}.desktop.performance`);

function isMissingChromeError(error) {
  return Boolean(
    error &&
    typeof error === "object" &&
    "message" in error &&
    typeof error.message === "string" &&
    (
      error.message.includes("CHROME_PATH environment variable") ||
      error.message.includes("Unable to find a valid Chrome installation")
    ),
  );
}

let chrome;
try {
  chrome = await launch({
    chromePath: process.env.CHROME_PATH,
    chromeFlags: ["--headless", "--disable-gpu", "--no-sandbox"],
  });

  await fs.mkdir(reportsDir, { recursive: true });

  const result = await lighthouse(targetUrl, {
    port: chrome.port,
    output: ["html", "json"],
    onlyCategories: ["performance"],
    preset: "desktop",
    logLevel: "info",
  });

  if (!result?.lhr || !result?.report) {
    throw new Error("Lighthouse did not return a report.");
  }

  const [htmlReport, jsonReport] = Array.isArray(result.report)
    ? result.report
    : [result.report, JSON.stringify(result.lhr, null, 2)];

  const htmlPath = `${reportStem}.html`;
  const jsonPath = `${reportStem}.json`;

  await fs.writeFile(htmlPath, htmlReport, "utf8");
  await fs.writeFile(jsonPath, jsonReport, "utf8");

  const score = result.lhr.categories.performance?.score;
  const percentScore = score === null || score === undefined
    ? "n/a"
    : `${Math.round(score * 100)}/100`;

  console.log(`Lighthouse performance score: ${percentScore}`);
  console.log(`Audited URL: ${result.lhr.finalDisplayedUrl ?? targetUrl}`);
  console.log(`HTML report: ${path.relative(projectRoot, htmlPath)}`);
  console.log(`JSON report: ${path.relative(projectRoot, jsonPath)}`);
} catch (error) {
  if (isMissingChromeError(error)) {
    console.error(
      "Chrome could not be found. Install Chrome/Chromium locally or set CHROME_PATH before running Lighthouse.",
    );
    process.exitCode = 1;
  } else {
    throw error;
  }
} finally {
  if (chrome) {
    await chrome.kill();
  }
}
