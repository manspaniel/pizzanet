# Repository Guidelines

## Project Structure & Module Organization

- `BRIEF.md` defines the product goal.
- `notes/UPDATED_PLAN.md` is the main architecture and implementation plan; detailed research lives beside it in `notes/`.
- `app/` contains the React 19, TypeScript, Vite, and Tailwind application. Put runtime code under `app/src/`, page components under `app/src/components/pages/`, public files under `app/public/`, and imported images under `app/src/assets/`.
- `samples/` contains visual reference images, not application assets.
- `references/` contains study-only upstream checkouts. Do not edit, reformat, or build them unless the task explicitly requires it.

Keep tracking/VIO, Burn inference, roof fitting, and Three.js rendering as separate modules exchanging timestamped data rather than shared graphics contexts.

## Build, Test, and Development Commands

Run frontend commands from `app/`:

```bash
pnpm install --frozen-lockfile  # install the locked dependencies
pnpm dev                       # start Vite with hot reload
pnpm lint                      # run Oxlint
pnpm build                     # type-check and create app/dist/
pnpm preview                   # serve the production build locally
```

No automated test runner is configured yet. At minimum, run `pnpm lint` and `pnpm build`, then manually exercise changed camera, permission, and UI paths on the relevant device.

## Coding Style & Naming Conventions

Use strict TypeScript and ES modules. Follow the existing style: two-space indentation, double quotes, semicolons, and trailing commas. Use PascalCase for React components and their files (`Home.tsx`), camelCase for functions and variables, and descriptive kebab-case names for static assets. Prefer small function components and typed boundaries between browser, WASM, inference, and rendering code. Oxlint configuration lives in `app/.oxlintrc.json`.

## Testing Guidelines

When a test framework is added, colocate unit tests as `*.test.ts` or `*.test.tsx`; keep recorded camera/sensor replays and integration fixtures outside production assets. Tests should cover timestamp association, geometry transforms, permission failures, tracking recovery, and stale-inference rejection. Never treat visual inspection alone as sufficient for geometry or synchronization logic.

## Commit & Pull Request Guidelines

The repository has no existing commits, so no historical convention exists. Use short imperative subjects, for example `Add motion permission flow`, and keep unrelated changes separate. Pull requests should include a concise summary, validation commands, linked issue or design note, and screenshots or video for visual changes. For device-specific AR work, name the tested iPhone/iOS version and permission state. Do not commit secrets, generated datasets, build output, or incidental `.DS_Store` files.
