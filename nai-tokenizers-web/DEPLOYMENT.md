# Deployment Guide

## GitHub Pages Setup

This project includes a GitHub Actions workflow that automatically builds and deploys the tokenizer demo to GitHub Pages.

### Prerequisites

1. **Enable GitHub Pages** in your repository settings:
   - Go to Settings → Pages
   - Under "Build and deployment", select "Source: GitHub Actions"

2. **Permissions** (should be set automatically by the workflow):
   - The workflow has `pages: write` and `id-token: write` permissions
   - These allow it to deploy to GitHub Pages

### Automatic Deployment

The workflow (`deploy-tokenizer-web.yml`) triggers on:
- ✅ Push to `main` branch
- ✅ Pull requests to `main` branch
- ✅ Manual workflow dispatch

### Build Process

1. **Build nai-tokenizers** with Brotli compression
   - Downloads tokenizer JSON files
   - Compresses them with Brotli quality 11
   - Embeds compressed data in the crate

2. **Build WASM module**
   - Compiles Rust to WebAssembly
   - Optimizes with `wasm-opt -Oz`
   - Enables bulk memory features
   - Final size: ~3.9 MB (includes compressed tokenizer)

3. **Deploy to GitHub Pages**
   - Uploads `nai-tokenizers-web/www/` directory
   - Deploys to `https://<username>.github.io/<repo>/`

### Manual Build

To build locally:

```bash
cd nai-tokenizers-web
npm run build        # Development build
npm run build:release # Release build with extra optimization
npm run serve        # Start local server
```

### Troubleshooting

**Build fails with "Device or resource busy"**
- This can happen on NFS mounts. The workflow uses GitHub's runners which don't have this issue.

**WASM fails to load**
- Make sure bulk memory is enabled in your browser (all modern browsers support it)
- Check browser console for errors

**Compression not working**
- Make sure brotli dependency is in both `[dependencies]` and `[build-dependencies]`
- Delete `nai-tokenizers/tokenizers/*.br` files to force recompression

### Size Optimization

The current build achieves:
- **84% size reduction** compared to uncompressed
- ~20 MB tokenizer → ~2 MB compressed → embedded in 3.9 MB WASM
- Decompression happens once at initialization with minimal overhead

### Cache

The workflow caches:
- Cargo registry and git dependencies
- Build artifacts (`target/` directories)
- wasm-pack binary

This speeds up subsequent builds significantly.