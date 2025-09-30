# nai-tokenizers-extism-host-go

Go host for the NAI tokenizers Extism plugin, providing both a library and CLI interface.

## Components

### Library (`tokenizer` package)

A Go library that wraps the Extism plugin with idiomatic Go APIs and proper error handling.

#### Single-threaded Use

For simple single-threaded applications:

```go
import "github.com/jpossaz/nai-tokenizers-extism-host-go/tokenizer"

tok, err := tokenizer.New("path/to/plugin.wasm")
if err != nil {
    log.Fatal(err)
}
defer tok.Close()

// Tokenize text
tokens, err := tok.Tokenize("Hello, world!", false)
if err != nil {
    log.Fatal(err)
}

// Detokenize tokens
text, err := tok.Detokenize(tokens, false)
if err != nil {
    log.Fatal(err)
}
```

#### Thread-safe / Concurrent Use

For production backends and concurrent use:

```go
import "github.com/jpossaz/nai-tokenizers-extism-host-go/tokenizer"

// Create a pooled tokenizer (thread-safe)
tok, err := tokenizer.NewPooled("path/to/plugin.wasm")
if err != nil {
    log.Fatal(err)
}
defer tok.Close()

// Use from multiple goroutines safely
var wg sync.WaitGroup
for i := 0; i < 100; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        tokens, err := tok.Tokenize("Hello, world!", false)
        if err != nil {
            log.Printf("Error: %v", err)
        }
    }()
}
wg.Wait()
```

The `PooledTokenizer` compiles the WASM once and creates instances on-demand from a pool, allowing safe concurrent access.

### CLI (`nai-tok`)

A command-line tool for tokenization and detokenization.

#### Build

```bash
go build -o nai-tok ./cmd/nai-tok
```

#### Usage

**Tokenize text:**
```bash
nai-tok -wasm plugin.wasm -mode tokenize "Hello, world!"
# Output: 123 456 789
```

**Detokenize tokens:**
```bash
nai-tok -wasm plugin.wasm -mode detokenize "123 456 789"
# Output: Hello, world!
```

**With special tokens:**
```bash
nai-tok -wasm plugin.wasm -mode tokenize -special "Hello, world!"
```

**JSON output:**
```bash
nai-tok -wasm plugin.wasm -mode tokenize -json "Hello, world!"
# Output: [123,456,789]
```

## Flags

- `-wasm <path>`: Path to the WASM plugin file (required)
- `-mode <mode>`: Operation mode: `tokenize` or `detokenize` (default: `tokenize`)
- `-special`: Include special tokens (default: `false`)
- `-json`: Output as JSON (default: `false`)
