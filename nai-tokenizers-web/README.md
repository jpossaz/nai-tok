# NovelAI GLM-4.5 Tokenizer (Web)

A high-quality WebAssembly tokenizer for NovelAI's GLM-4.5 model, featuring a web worker API for non-blocking tokenization and a beautiful interactive demo.

## Features

- 🚀 **Fast**: Runs in a Web Worker for non-blocking performance
- 🎨 **Beautiful UI**: Modern, responsive interface with live token highlighting
- 🔤 **Token Visualization**: See tokens highlighted with colors and hover tooltips
- 📊 **Statistics**: Real-time character count, token count, and char/token ratio
- ⚡ **Real-time**: Tokenizes as you type with debouncing
- 🎯 **Special Tokens**: Support for special tokens like `[gMASK]`, `<|system|>`, etc.

## Quick Start

### Prerequisites

- Rust (with `wasm-pack`)
- Node.js (for the dev server)
- Python 3 (alternative for serving)

### Building

```bash
# Build the WASM module
npm run build

# Or build optimized release version
npm run build:release
```

### Running

```bash
# Start development server (builds and serves)
npm run dev

# Or just serve (after building)
npm run serve
```

Then open http://localhost:8080 in your browser.

## API Usage

### In a Web Worker (Recommended)

```javascript
// worker.js
import init, { tokenize, detokenize } from './pkg/nai_tokenizers_web.js';

await init();

const result = tokenize("Hello, world!", true);
console.log(result);
// {
//   tokens: [
//     { id: 9707, text: "Hello", start: 0, end: 5 },
//     { id: 11, text: ",", start: 5, end: 6 },
//     { id: 1879, text: " world", start: 6, end: 12 },
//     { id: 0, text: "!", start: 12, end: 13 }
//   ],
//   ids: [9707, 11, 1879, 0]
// }
```

### Direct API

```javascript
import init, { tokenize, detokenize, get_tokenizer_info } from './pkg/nai_tokenizers_web.js';

// Initialize WASM module
await init();

// Tokenize text
const result = tokenize("Hello, world!", true);
console.log(result.tokens);
console.log(result.ids);

// Detokenize
const text = detokenize([9707, 11, 1879, 0], true);
console.log(text); // "Hello, world!"

// Get tokenizer info
const info = get_tokenizer_info();
console.log(info);
```

### Using the Worker Wrapper

The included `TokenizerWorker` class provides a clean Promise-based API:

```javascript
import { TokenizerWorker } from './tokenizer-worker.js';

const worker = new TokenizerWorker('./worker.js');

// Tokenize
const result = await worker.tokenize("Hello, world!", true);
console.log(result);

// Detokenize
const decoded = await worker.detokenize([9707, 11, 1879, 0], true);
console.log(decoded.text);

// Get info
const info = await worker.getInfo();
console.log(info);
```

## Functions

### `tokenize(text: string, keep_special_tokens: bool) -> TokenizeResult`

Tokenizes input text and returns detailed information about each token.

**Parameters:**
- `text`: The input text to tokenize
- `keep_special_tokens`: Whether to preserve special tokens (e.g., `[gMASK]`, `<|system|>`)

**Returns:**
```typescript
{
  tokens: Array<{
    id: number,
    text: string,
    start: number,
    end: number
  }>,
  ids: number[]
}
```

### `detokenize(ids: number[], keep_special_tokens: bool) -> string`

Decodes token IDs back to text.

**Parameters:**
- `ids`: Array of token IDs
- `keep_special_tokens`: Whether to preserve special tokens in output

**Returns:** The decoded text string

### `get_tokenizer_info() -> object`

Returns information about the tokenizer.

## Architecture

```
nai-tokenizers-web/
├── src/
│   └── lib.rs              # Rust WASM bindings
├── www/
│   ├── index.html          # Interactive demo
│   ├── worker.js           # Web Worker implementation
│   └── pkg/                # Built WASM artifacts (generated)
├── Cargo.toml
└── package.json
```

## Performance

The tokenizer runs in a Web Worker, keeping the main thread free for UI interactions. Tokenization is fast and non-blocking, making it suitable for real-time applications.

## Special Tokens

The tokenizer supports GLM-4.5 special tokens:
- `[gMASK]` - Generation mask token
- `<|system|>` - System message delimiter
- `<|user|>` - User message delimiter
- `<|assistant|>` - Assistant message delimiter
- And more...

## License

MIT