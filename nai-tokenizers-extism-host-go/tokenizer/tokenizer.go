package tokenizer

import (
	"context"
	"fmt"

	extism "github.com/extism/go-sdk"
	"github.com/vmihailenco/msgpack/v5"
)

// Tokenizer wraps an Extism plugin for tokenization operations.
// Note: This is NOT thread-safe. For concurrent use, use PooledTokenizer instead.
type Tokenizer struct {
	plugin *extism.Plugin
}

// TokenizeInput represents the input for tokenization
type TokenizeInput struct {
	Text                  string `msgpack:"text"`
	IncludeSpecialTokens bool   `msgpack:"include_special_tokens"`
}

// DetokenizeInput represents the input for detokenization
type DetokenizeInput struct {
	Tokens                []uint32 `msgpack:"tokens"`
	IncludeSpecialTokens bool     `msgpack:"include_special_tokens"`
}

// New creates a new Tokenizer from a WASM file path.
// Note: This is NOT thread-safe. For concurrent use, use NewPooled instead.
func New(wasmPath string) (*Tokenizer, error) {
	ctx := context.Background()
	manifest := extism.Manifest{
		Wasm: []extism.Wasm{
			extism.WasmFile{
				Path: wasmPath,
			},
		},
	}

	config := extism.PluginConfig{
		EnableWasi: true,
	}

	plugin, err := extism.NewPlugin(ctx, manifest, config, []extism.HostFunction{})
	if err != nil {
		return nil, fmt.Errorf("failed to create plugin: %w", err)
	}

	return &Tokenizer{plugin: plugin}, nil
}

// Tokenize converts text to tokens
func (t *Tokenizer) Tokenize(text string, includeSpecialTokens bool) ([]uint32, error) {
	input := TokenizeInput{
		Text:                  text,
		IncludeSpecialTokens: includeSpecialTokens,
	}

	inputData, err := msgpack.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	_, output, err := t.plugin.Call("tokenize", inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to call tokenize: %w", err)
	}

	var tokens []uint32
	if err := msgpack.Unmarshal(output, &tokens); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return tokens, nil
}

// Detokenize converts tokens to text
func (t *Tokenizer) Detokenize(tokens []uint32, includeSpecialTokens bool) (string, error) {
	input := DetokenizeInput{
		Tokens:                tokens,
		IncludeSpecialTokens: includeSpecialTokens,
	}

	inputData, err := msgpack.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("failed to marshal input: %w", err)
	}

	_, output, err := t.plugin.Call("detokenize", inputData)
	if err != nil {
		return "", fmt.Errorf("failed to call detokenize: %w", err)
	}

	return string(output), nil
}

// Close releases the plugin resources
func (t *Tokenizer) Close() error {
	if t.plugin != nil {
		return t.plugin.Close(context.Background())
	}
	return nil
}
