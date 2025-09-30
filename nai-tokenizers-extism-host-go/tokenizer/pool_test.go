package tokenizer

import (
	"os"
	"sync"
	"testing"
)

func TestPooledTokenizer_New(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := NewPooled(wasmPath)
	if err != nil {
		t.Fatalf("failed to create pooled tokenizer: %v", err)
	}
	defer tok.Close()

	if tok == nil {
		t.Fatal("pooled tokenizer is nil")
	}
	if tok.compiled == nil {
		t.Fatal("compiled plugin is nil")
	}
}

func TestPooledTokenizer_Tokenize(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := NewPooled(wasmPath)
	if err != nil {
		t.Fatalf("failed to create pooled tokenizer: %v", err)
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

func TestPooledTokenizer_Detokenize(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := NewPooled(wasmPath)
	if err != nil {
		t.Fatalf("failed to create pooled tokenizer: %v", err)
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

func TestPooledTokenizer_RoundTrip(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := NewPooled(wasmPath)
	if err != nil {
		t.Fatalf("failed to create pooled tokenizer: %v", err)
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

func TestPooledTokenizer_Concurrent(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := NewPooled(wasmPath)
	if err != nil {
		t.Fatalf("failed to create pooled tokenizer: %v", err)
	}
	defer tok.Close()

	const numGoroutines = 50
	const numIterations = 10

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*numIterations)

	// Test concurrent tokenization
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				text := "Hello, world!"
				tokens, err := tok.Tokenize(text, false)
				if err != nil {
					errors <- err
					return
				}
				if len(tokens) == 0 {
					errors <- err
					return
				}
			}
		}(i)
	}

	// Test concurrent detokenization
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				tokens := []uint32{9703, 11, 1879, 0}
				text, err := tok.Detokenize(tokens, false)
				if err != nil {
					errors <- err
					return
				}
				if text != "Hello, world!" {
					errors <- err
					return
				}
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		if err != nil {
			t.Errorf("concurrent operation error: %v", err)
		}
	}
}

func TestPooledTokenizer_ConcurrentRoundTrip(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := NewPooled(wasmPath)
	if err != nil {
		t.Fatalf("failed to create pooled tokenizer: %v", err)
	}
	defer tok.Close()

	const numGoroutines = 50
	const numIterations = 10

	texts := []string{
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog.",
		"Testing concurrent access.",
		"Another test string.",
		"Go is awesome!",
	}

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*numIterations)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				text := texts[j%len(texts)]

				// Tokenize
				tokens, err := tok.Tokenize(text, false)
				if err != nil {
					errors <- err
					return
				}

				// Detokenize
				result, err := tok.Detokenize(tokens, false)
				if err != nil {
					errors <- err
					return
				}

				// Verify
				if result != text {
					errors <- err
					return
				}
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		if err != nil {
			t.Errorf("concurrent round-trip error: %v", err)
		}
	}
}

func TestPooledTokenizer_Close(t *testing.T) {
	wasmPath := getTestWasmPath(t)

	tok, err := NewPooled(wasmPath)
	if err != nil {
		t.Fatalf("failed to create pooled tokenizer: %v", err)
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

func BenchmarkPooledTokenizer_Tokenize(b *testing.B) {
	wasmPath := testWasmPath
	if _, err := os.Stat(wasmPath); os.IsNotExist(err) {
		b.Skipf("WASM file not found at %s", wasmPath)
	}

	tok, err := NewPooled(wasmPath)
	if err != nil {
		b.Fatalf("failed to create pooled tokenizer: %v", err)
	}
	defer tok.Close()

	text := "Hello, world! This is a benchmark test."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tok.Tokenize(text, false)
		if err != nil {
			b.Fatalf("Tokenize() error = %v", err)
		}
	}
}

func BenchmarkPooledTokenizer_TokenizeParallel(b *testing.B) {
	wasmPath := testWasmPath
	if _, err := os.Stat(wasmPath); os.IsNotExist(err) {
		b.Skipf("WASM file not found at %s", wasmPath)
	}

	tok, err := NewPooled(wasmPath)
	if err != nil {
		b.Fatalf("failed to create pooled tokenizer: %v", err)
	}
	defer tok.Close()

	text := "Hello, world! This is a benchmark test."

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := tok.Tokenize(text, false)
			if err != nil {
				b.Fatalf("Tokenize() error = %v", err)
			}
		}
	})
}
