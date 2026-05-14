# rbf-raymarching

Minimal WebGPU starter built with TypeScript and webpack.

## Local setup

1. Install Node.js 22 or newer.
2. Install dependencies:

```bash
npm install
```

3. Start the dev server:

```bash
npm run dev
```

4. Open `http://localhost:8080`.

## Build

Create a production bundle in `dist/`:

```bash
npm run build
```

## Deploy to GitHub Pages

1. Push this repo to GitHub.
2. In the repository settings, open `Pages`.
3. Set the source to `GitHub Actions`.
4. Push to `main`.
5. Wait for the `Deploy GitHub Pages` workflow to finish.
6. Open the URL shown in the deployment job.

The workflow uses the GitHub Pages base path automatically, so it works for project pages such as `https://username.github.io/rbf-raymarching/`.

## Vercel alternative

This project also works as a simple static Vercel deployment:

1. Import the repository into Vercel.
2. Set the build command to `npm run build`.
3. Set the output directory to `dist`.
4. Deploy.

Vercel is useful if you want preview links for branches and pull requests. GitHub Pages is simpler if you only need one stable URL from GitHub.
