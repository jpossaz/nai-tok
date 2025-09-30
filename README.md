# nai-tok

High-performance tokenizer implementation for NovelAI's GLM-4.5 model, with comprehensive bindings for multiple platforms and use cases.

## Motivation

Language model tokenization is a critical component in AI applications, but existing solutions often fall short in several key areas:

1. **Performance**: Many tokenizer implementations are slow, especially when used from non-Python languages or in web environments
2. **Portability**: Cross-platform deployment requires managing multiple dependencies and language-specific implementations
3. **Consistency**: Different implementations can produce subtly different results, causing issues in production
4. **Chat Templating**: Most tokenizers don't include built-in support for chat message formatting, requiring manual string manipulation that's error-prone

This project solves these problems by providing:

- **Single Source of Truth**: Core tokenization logic written once in Rust
- **Zero-Copy Architecture**: Efficient memory usage and blazing fast performance
- **Universal Deployment**: Compile to native binaries, WebAssembly, or Extism plugins
- **Chat Template Support**: Built-in OpenAI-compatible chat formatting with reasoning support
- **Type Safety**: Strong typing in both Rust and Go eliminates entire classes of bugs

## Project Structure

```
nai-tok/
├── nai-tokenizers/              # Core Rust library
│   ├── src/
│   │   └── lib.rs              # Tokenizer + chat template implementation
│   └── tests/                   # Comprehensive snapshot tests
├── nai-tokenizers-web/          # WebAssembly bindings
│   ├── src/lib.rs              # wasm-bindgen interface
│   └── www/                     # Interactive web demo
├── nai-tokenizers-extism/       # Extism plugin
│   └── src/lib.rs              # Plugin with chat template support
└── nai-tokenizers-extism-host-go/  # Go host library
    ├── tokenizer/               # Go library with type-safe API
    └── cmd/nai-tok/            # CLI tool
```

## Components

### 1. Core Library (`nai-tokenizers`)

The foundational Rust library containing:
- GLM-4.5 tokenizer with Brotli-compressed vocabulary (saves ~10MB per deployment)
- Chat templating engine with reasoning support
- Zero external dependencies for WASM builds

**Features:**
- Fast tokenization/detokenization
- Special token handling
- OpenAI-compatible message formatting
- Reasoning/thinking tag support
- Configurable prefills for different inference scenarios

### 2. WebAssembly (`nai-tokenizers-web`)

Browser-ready tokenizer with an interactive demo interface.

**Use Cases:**
- Client-side token counting for UI/UX
- Real-time token visualization during text editing
- Offline-capable web applications
- Browser extensions

**Features:**
- Small compressed WASM bundle (including vocabulary)
- Web Worker support for non-blocking tokenization
- Interactive demo with token visualization
- TypeScript-ready API

[See detailed documentation →](./nai-tokenizers-web/README.md)

### 3. Extism Plugin (`nai-tokenizers-extism`)

Universal plugin format that runs anywhere: CLI, serverless functions, microservices, or embedded in other languages.

**Why Extism?**
- **Language Agnostic**: Call from Go, Python, Node.js, Ruby, etc.
- **Sandboxed**: Safe to run untrusted code
- **Portable**: Single WASM file runs everywhere
- **Fast**: Near-native performance with minimal overhead

**Features:**
- Tokenize/detokenize operations
- Full chat template support with OpenAI compatibility
- MessagePack serialization for efficiency
- Reasoning and prefill control

### 4. Go Host Library (`nai-tokenizers-extism-host-go`)

Idiomatic Go library and CLI tool for using the tokenizer.

**Use Cases:**
- Backend services and APIs
- CLI tools and scripts
- Batch processing pipelines
- Integration with existing Go codebases

**Features:**
- Thread-safe pooled tokenizer for concurrent use
- Type-safe API with helper functions
- Comprehensive error handling
- Command-line interface with JSON support
- Benchmark mode for performance testing

[See detailed documentation →](./nai-tokenizers-extism-host-go/README.md)

## Chat Templating

One of the key features is built-in support for chat message formatting. The implementation handles:

- **OpenAI-Compatible Format**: Standard `role`/`content` message structure
- **Reasoning Support**: Optional `reasoning_content` field for chain-of-thought
- **Flexible Prefills**: Control how prompts end (none, canonical, partial reasoning, full reasoning)
- **Position Awareness**: Automatically handles intermediate vs. last message formatting
- **Special Tokens**: Proper insertion of model-specific control tokens

Example message format:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "reasoning_enabled": true,
  "prefill": {"type": "canonical"}
}
```

This eliminates the need for manual string manipulation and ensures consistency across different deployment environments.

## Getting Started

### Web Demo

```bash
cd nai-tokenizers-web
npm install
npm run build
npm run serve
```

Open http://localhost:8080 to see the interactive tokenizer.

### Go CLI

```bash
cd nai-tokenizers-extism-host-go
go build -o nai-tok ./cmd/nai-tok

# Tokenize text
./nai-tok -wasm ../target/wasm32-unknown-unknown/release/nai_tokenizers_extism.wasm \
  -mode tokenize "Hello, world!"

# Apply chat template
./nai-tok -wasm ../target/wasm32-unknown-unknown/release/nai_tokenizers_extism.wasm \
  -mode chat -chat-file chat.json
```

### Building from Source

```bash
# Build core library
cd nai-tokenizers
cargo build --release

# Build WASM for web
cd ../nai-tokenizers-web
npm run build

# Build Extism plugin
cd ../nai-tokenizers-extism
cargo build --target wasm32-unknown-unknown --release

# Build Go CLI
cd ../nai-tokenizers-extism-host-go
go build ./cmd/nai-tok
```

## Use Cases

### Web Applications
- Token counting for input validation
- Real-time token visualization
- Client-side cost estimation
- Offline-capable editors

### Backend Services
- API rate limiting by tokens
- Batch text processing
- Chat message formatting for LLM APIs
- Training data preparation

### CLI Tools
- Text analysis scripts
- Data preprocessing pipelines
- Token statistics for large datasets
- Format conversion utilities

### Embedded Applications
- Browser extensions
- Desktop applications (via WebView)
- Mobile apps (via React Native, etc.)
- IoT devices with WASM support

## Testing

The project includes comprehensive snapshot tests to ensure correctness:

```bash
cd nai-tokenizers
cargo test --features glm45_template
```

All chat templating scenarios are covered, including:
- Basic conversations
- Multi-turn dialogues
- Reasoning content handling
- All prefill types
- Edge cases (empty messages, etc.)

## License

MIT

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. Add snapshot tests for new template features
4. Update relevant READMEs

## Acknowledgments

- Built for NovelAI's GLM-4.5 model
- Uses the Hugging Face tokenizers library
- WASM tooling powered by wasm-pack and wasm-bindgen
- Plugin system powered by Extism
