package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

const testWasmPath = "../../target/wasm32-unknown-unknown/debug/nai_tokenizers_extism.wasm"

func getTestWasmPath(t *testing.T) string {
	path := testWasmPath
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("WASM file not found at %s - run 'cargo build --target wasm32-unknown-unknown' first", path)
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		t.Fatalf("failed to get absolute path: %v", err)
	}
	return absPath
}

func TestTokenizer_New(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := New(wasmPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}
	defer tok.Close()

	if tok == nil {
		t.Fatal("tokenizer is nil")
	}
	if tok.plugin == nil {
		t.Fatal("plugin is nil")
	}
}

func TestTokenizer_Tokenize(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := New(wasmPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}
	defer tok.Close()

	tests := []struct {
		name                 string
		text                 string
		includeSpecialTokens bool
		wantEmpty            bool
	}{
		{
			name:                 "simple text",
			text:                 "Hello, world!",
			includeSpecialTokens: false,
			wantEmpty:            false,
		},
		{
			name:                 "empty text",
			text:                 "",
			includeSpecialTokens: false,
			wantEmpty:            true,
		},
		{
			name:                 "with special tokens",
			text:                 "Hello, world!",
			includeSpecialTokens: true,
			wantEmpty:            false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := tok.Tokenize(tt.text, tt.includeSpecialTokens)
			if err != nil {
				t.Fatalf("Tokenize() error = %v", err)
			}
			if tt.wantEmpty && len(tokens) != 0 {
				t.Errorf("Tokenize() expected empty tokens, got %d tokens", len(tokens))
			}
			if !tt.wantEmpty && len(tokens) == 0 {
				t.Errorf("Tokenize() expected non-empty tokens, got empty")
			}
		})
	}
}

func TestTokenizer_Detokenize(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := New(wasmPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}
	defer tok.Close()

	tests := []struct {
		name                 string
		tokens               []uint32
		includeSpecialTokens bool
		wantEmpty            bool
	}{
		{
			name:                 "simple tokens",
			tokens:               []uint32{9703, 11, 1879, 0},
			includeSpecialTokens: false,
			wantEmpty:            false,
		},
		{
			name:                 "empty tokens",
			tokens:               []uint32{},
			includeSpecialTokens: false,
			wantEmpty:            true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, err := tok.Detokenize(tt.tokens, tt.includeSpecialTokens)
			if err != nil {
				t.Fatalf("Detokenize() error = %v", err)
			}
			if tt.wantEmpty && text != "" {
				t.Errorf("Detokenize() expected empty text, got %q", text)
			}
			if !tt.wantEmpty && text == "" {
				t.Errorf("Detokenize() expected non-empty text, got empty")
			}
		})
	}
}

func TestTokenizer_RoundTrip(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := New(wasmPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}
	defer tok.Close()

	tests := []struct {
		name                 string
		text                 string
		includeSpecialTokens bool
	}{
		{
			name:                 "hello world",
			text:                 "Hello, world!",
			includeSpecialTokens: false,
		},
		{
			name:                 "complex text",
			text:                 "The quick brown fox jumps over the lazy dog.",
			includeSpecialTokens: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Tokenize
			tokens, err := tok.Tokenize(tt.text, tt.includeSpecialTokens)
			if err != nil {
				t.Fatalf("Tokenize() error = %v", err)
			}

			// Detokenize
			result, err := tok.Detokenize(tokens, tt.includeSpecialTokens)
			if err != nil {
				t.Fatalf("Detokenize() error = %v", err)
			}

			// Compare
			if result != tt.text {
				t.Errorf("RoundTrip mismatch:\noriginal: %q\nresult:   %q", tt.text, result)
			}
		})
	}
}

func TestTokenizer_Close(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := New(wasmPath)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}

	err = tok.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}

	// Calling Close again should not panic
	err = tok.Close()
	if err != nil {
		t.Errorf("Close() second call error = %v", err)
	}
}
