package tokenizer

import (
	"context"
	"fmt"
	"sync"

	extism "github.com/extism/go-sdk"
	"github.com/vmihailenco/msgpack/v5"
)

// PooledTokenizer is a thread-safe tokenizer that uses a pool of plugin instances.
// It compiles the WASM once and creates instances on-demand for concurrent use.
type PooledTokenizer struct {
	compiled *extism.CompiledPlugin
	pool     sync.Pool
	mu       sync.Mutex
}

// NewPooled creates a new thread-safe PooledTokenizer from a WASM file path.
// This tokenizer can be safely used across multiple goroutines.
func NewPooled(wasmPath string) (*PooledTokenizer, error) {
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

	compiled, err := extism.NewCompiledPlugin(ctx, manifest, config, []extism.HostFunction{})
	if err != nil {
		return nil, fmt.Errorf("failed to compile plugin: %w", err)
	}

	pt := &PooledTokenizer{
		compiled: compiled,
	}

	pt.pool.New = func() interface{} {
		ctx := context.Background()
		instance, err := compiled.Instance(ctx, extism.PluginInstanceConfig{})
		if err != nil {
			// This shouldn't normally fail, but we need to handle it
			return nil
		}
		return instance
	}

	return pt, nil
}

// getInstance gets a plugin instance from the pool
func (pt *PooledTokenizer) getInstance() (*extism.Plugin, error) {
	instance := pt.pool.Get()
	if instance == nil {
		return nil, fmt.Errorf("failed to get plugin instance from pool")
	}
	plugin, ok := instance.(*extism.Plugin)
	if !ok || plugin == nil {
		return nil, fmt.Errorf("invalid plugin instance in pool")
	}
	return plugin, nil
}

// putInstance returns a plugin instance to the pool
func (pt *PooledTokenizer) putInstance(plugin *extism.Plugin) {
	pt.pool.Put(plugin)
}

// Tokenize converts text to tokens. Thread-safe.
func (pt *PooledTokenizer) Tokenize(text string, includeSpecialTokens bool) ([]uint32, error) {
	plugin, err := pt.getInstance()
	if err != nil {
		return nil, err
	}
	defer pt.putInstance(plugin)

	input := TokenizeInput{
		Text:                  text,
		IncludeSpecialTokens: includeSpecialTokens,
	}

	inputData, err := msgpack.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	_, output, err := plugin.Call("tokenize", inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to call tokenize: %w", err)
	}

	var tokens []uint32
	if err := msgpack.Unmarshal(output, &tokens); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return tokens, nil
}

// Detokenize converts tokens to text. Thread-safe.
func (pt *PooledTokenizer) Detokenize(tokens []uint32, includeSpecialTokens bool) (string, error) {
	plugin, err := pt.getInstance()
	if err != nil {
		return "", err
	}
	defer pt.putInstance(plugin)

	input := DetokenizeInput{
		Tokens:                tokens,
		IncludeSpecialTokens: includeSpecialTokens,
	}

	inputData, err := msgpack.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("failed to marshal input: %w", err)
	}

	_, output, err := plugin.Call("detokenize", inputData)
	if err != nil {
		return "", fmt.Errorf("failed to call detokenize: %w", err)
	}

	return string(output), nil
}

// Close releases all plugin resources. Should be called when done using the tokenizer.
func (pt *PooledTokenizer) Close() error {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if pt.compiled != nil {
		return pt.compiled.Close(context.Background())
	}
	return nil
}
