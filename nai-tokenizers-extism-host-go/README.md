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

// Apply chat template
result, err := tok.ChatTemplate(tokenizer.ChatTemplateInput{
    Messages: []tokenizer.Message{
        {Role: "system", Content: "You are a helpful assistant."},
        {Role: "user", Content: "Hello!"},
    },
    ReasoningEnabled: false,
    Prefill:          tokenizer.PrefillCanonical(),
})
if err != nil {
    log.Fatal(err)
}

// With partial reasoning prefill
reasoningResult, err := tok.ChatTemplate(tokenizer.ChatTemplateInput{
    Messages: []tokenizer.Message{
        {Role: "user", Content: "Is 97 prime?"},
    },
    ReasoningEnabled: true,
    Prefill:          tokenizer.PrefillPartialReasoning("Let me check..."),
})
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

A command-line tool for tokenization, detokenization, and chat templating.

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

**Chat template (from file):**
```bash
nai-tok -wasm plugin.wasm -mode chat -chat-file chat.json
```

**Chat template (from command line):**
```bash
nai-tok -wasm plugin.wasm -mode chat '{"messages":[{"role":"user","content":"Hello!"}],"reasoning_enabled":true}'
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

**Benchmark mode:**
```bash
nai-tok -wasm plugin.wasm -mode chat -chat-file chat.json -benchmark
```

## Flags

- `-wasm <path>`: Path to the WASM plugin file (required)
- `-mode <mode>`: Operation mode: `tokenize`, `detokenize`, or `chat` (default: `tokenize`)
- `-special`: Include special tokens (default: `false`)
- `-json`: Output as JSON (default: `false`)
- `-chat-file <path>`: Path to JSON file with chat messages (for chat mode)
- `-benchmark`: Show timing information (default: `false`)

## Chat Template

The chat template mode supports OpenAI-compatible message formats with additional options for reasoning and prefill control.

### Message Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is 2+2?"
    },
    {
      "role": "assistant",
      "content": "The answer is 4.",
      "reasoning_content": "2 + 2 = 4"
    }
  ],
  "reasoning_enabled": true,
  "ignore_message_position": false,
  "prefill": {
    "type": "canonical"
  }
}
```

### Supported Roles

- `system` or `developer`: System instructions
- `user`: User messages
- `assistant`: Assistant responses (can include `reasoning_content`)

### Prefill Types

The `prefill` field controls how the template ends:

**None** - No prefill, just render messages:
```json
{
  "prefill": {
    "type": "none"
  }
}
```

**Canonical** - Standard prefill with assistant sentinel (default):
```json
{
  "prefill": {
    "type": "canonical"
  }
}
```

**Partial Reasoning** - Start with reasoning tag and content:
```json
{
  "prefill": {
    "type": "partial_reasoning",
    "reasoning_content": "Let me think..."
  }
}
```

**Full Reasoning** - Include complete reasoning and start of response:
```json
{
  "prefill": {
    "type": "full_reasoning",
    "reasoning_content": "Let me calculate: 2+2=4",
    "content": "The answer is"
  }
}
```

### Options

- `reasoning_enabled` (bool): Whether reasoning mode is enabled for the model. Affects how prefills are rendered.
- `ignore_message_position` (bool): If `true`, treats all messages as if they were the last message. This causes reasoning content in intermediate assistant messages to be rendered. Useful for training data or preserving full conversation history. Default: `false`.

### Go Helper Functions

For type-safe prefill creation in Go code:

```go
// No prefill
tokenizer.PrefillNone()

// Canonical prefill (default)
tokenizer.PrefillCanonical()

// Partial reasoning
tokenizer.PrefillPartialReasoning("Let me think...")

// Full reasoning
tokenizer.PrefillFullReasoning("Calculation: 2+2=4", "The answer is")
```

Example usage:
```go
result, err := tok.ChatTemplate(tokenizer.ChatTemplateInput{
    Messages: []tokenizer.Message{
        {Role: "user", Content: "What is 2+2?"},
    },
    ReasoningEnabled: true,
    Prefill:          tokenizer.PrefillFullReasoning("2+2=4", "The answer is"),
})
```

### Examples

**Basic conversation:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "reasoning_enabled": false
}
```

**Multi-turn with reasoning:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "Is 17 prime?"},
    {"role": "assistant", "content": "Yes, 17 is prime."},
    {"role": "user", "content": "What about 21?"}
  ],
  "reasoning_enabled": true
}
```

**Preserve all reasoning (data export):**
```json
{
  "messages": [
    {"role": "user", "content": "Calculate 15 * 24"},
    {
      "role": "assistant",
      "content": "The answer is 360.",
      "reasoning_content": "15 * 24 = 15 * (20 + 4) = 300 + 60 = 360"
    }
  ],
  "reasoning_enabled": true,
  "ignore_message_position": true
}
```
