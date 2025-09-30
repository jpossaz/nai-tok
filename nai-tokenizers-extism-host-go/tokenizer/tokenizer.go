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

// Message represents an OpenAI-compatible chat message
type Message struct {
	Role             string  `json:"role" msgpack:"role"`
	Content          string  `json:"content" msgpack:"content"`
	ReasoningContent *string `json:"reasoning_content,omitempty" msgpack:"reasoning_content,omitempty"`
}

// PrefillType represents the type of prefill to use
type PrefillType struct {
	Type             string  `json:"type" msgpack:"type"`
	ReasoningContent *string `json:"reasoning_content,omitempty" msgpack:"reasoning_content,omitempty"`
	Content          *string `json:"content,omitempty" msgpack:"content,omitempty"`
}

// Helper functions for creating PrefillType instances

// PrefillNone creates a prefill with no additional content
func PrefillNone() *PrefillType {
	return &PrefillType{Type: "none"}
}

// PrefillCanonical creates a canonical prefill (default)
func PrefillCanonical() *PrefillType {
	return &PrefillType{Type: "canonical"}
}

// PrefillPartialReasoning creates a prefill with partial reasoning content
func PrefillPartialReasoning(reasoningContent string) *PrefillType {
	return &PrefillType{
		Type:             "partial_reasoning",
		ReasoningContent: &reasoningContent,
	}
}

// PrefillFullReasoning creates a prefill with full reasoning and content
func PrefillFullReasoning(reasoningContent, content string) *PrefillType {
	return &PrefillType{
		Type:             "full_reasoning",
		ReasoningContent: &reasoningContent,
		Content:          &content,
	}
}

// ChatTemplateInput represents the input for chat templating
type ChatTemplateInput struct {
	Messages               []Message    `json:"messages" msgpack:"messages"`
	ReasoningEnabled       bool         `json:"reasoning_enabled,omitempty" msgpack:"reasoning_enabled"`
	Prefill                *PrefillType `json:"prefill,omitempty" msgpack:"prefill,omitempty"`
	IgnoreMessagePosition  bool         `json:"ignore_message_position,omitempty" msgpack:"ignore_message_position"`
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

// ChatTemplate applies chat templating to messages
func (t *Tokenizer) ChatTemplate(input ChatTemplateInput) (string, error) {
	inputData, err := msgpack.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("failed to marshal input: %w", err)
	}

	_, output, err := t.plugin.Call("chat_template", inputData)
	if err != nil {
		return "", fmt.Errorf("failed to call chat_template: %w", err)
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
