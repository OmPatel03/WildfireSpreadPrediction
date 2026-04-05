# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.


## Build and Run

### For Development
1. cd to `frontend/wildfireFE`
2. run `npm install`
3. create `frontend/wildfireFE/.env` and declare the `MAPBOX` token
4. run `npm run dev`

Frontend builds and tests require Node `^20.19.0 || ^22.12.0 || >=24.0.0`.

## Lighthouse Performance Audits

This frontend includes Lighthouse-based performance tests for the initial page load.

### Available commands
- `npm run perf:lighthouse:local`
  - Builds the app, starts a local production preview on `http://127.0.0.1:4173`, and runs Lighthouse against it.
- `npm run perf:lighthouse:live`
  - Runs Lighthouse against the deployed site at `https://wisprlabs.com`.
- `npm run perf:lighthouse`
  - Runs both the local and live audits sequentially.

### Report output

Generated reports are written to `frontend/wildfireFE/lighthouse-reports/` as:
- one HTML report
- one JSON report

The current setup uses Lighthouse's desktop profile and audits only the `performance` category for the root page load.

### Browser requirement

Lighthouse needs a locally installed Chrome or Chromium browser. If it is not auto-detected, set `CHROME_PATH` before running the audit commands.
