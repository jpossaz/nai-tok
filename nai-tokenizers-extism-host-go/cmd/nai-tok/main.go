package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jpossaz/nai-tokenizers-extism-host-go/tokenizer"
)

func main() {
	var (
		wasmPath             = flag.String("wasm", "", "Path to WASM file (required)")
		mode                 = flag.String("mode", "tokenize", "Mode: 'tokenize', 'detokenize', or 'chat'")
		includeSpecialTokens = flag.Bool("special", false, "Include special tokens")
		jsonOutput           = flag.Bool("json", false, "Output as JSON")
		benchmark            = flag.Bool("benchmark", false, "Show timing information")
		chatFile             = flag.String("chat-file", "", "Path to JSON file with chat messages (for chat mode)")
	)

	flag.Parse()

	if *wasmPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -wasm flag is required")
		flag.Usage()
		os.Exit(1)
	}

	// Input validation depends on mode
	if *mode != "chat" && flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input is required")
		flag.Usage()
		os.Exit(1)
	}

	input := strings.Join(flag.Args(), " ")

	startLoad := time.Now()
	tok, err := tokenizer.New(*wasmPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating tokenizer: %v\n", err)
		os.Exit(1)
	}
	defer tok.Close()
	loadDuration := time.Since(startLoad)

	if *benchmark {
		fmt.Fprintf(os.Stderr, "Load time: %v\n", loadDuration)
	}

	switch *mode {
	case "tokenize":
		startOp := time.Now()
		tokens, err := tok.Tokenize(input, *includeSpecialTokens)
		opDuration := time.Since(startOp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error tokenizing: %v\n", err)
			os.Exit(1)
		}

		if *benchmark {
			fmt.Fprintf(os.Stderr, "Tokenize time (1st): %v\n", opDuration)

			// Second call to measure warmup
			startOp2 := time.Now()
			_, err := tok.Tokenize(input, *includeSpecialTokens)
			opDuration2 := time.Since(startOp2)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error tokenizing (2nd): %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "Tokenize time (2nd): %v\n", opDuration2)
			fmt.Fprintf(os.Stderr, "Total time: %v\n", loadDuration+opDuration+opDuration2)
		}

		if *jsonOutput {
			json.NewEncoder(os.Stdout).Encode(tokens)
		} else {
			for i, token := range tokens {
				if i > 0 {
					fmt.Print(" ")
				}
				fmt.Print(token)
			}
			fmt.Println()
		}

	case "detokenize":
		// Parse tokens from input
		startParse := time.Now()
		tokenStrs := strings.Fields(input)
		tokens := make([]uint32, 0, len(tokenStrs))
		for _, tokenStr := range tokenStrs {
			token, err := strconv.ParseUint(tokenStr, 10, 32)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error parsing token '%s': %v\n", tokenStr, err)
				os.Exit(1)
			}
			tokens = append(tokens, uint32(token))
		}
		parseDuration := time.Since(startParse)

		startOp := time.Now()
		text, err := tok.Detokenize(tokens, *includeSpecialTokens)
		opDuration := time.Since(startOp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error detokenizing: %v\n", err)
			os.Exit(1)
		}

		if *benchmark {
			fmt.Fprintf(os.Stderr, "Parse time: %v\n", parseDuration)
			fmt.Fprintf(os.Stderr, "Detokenize time: %v\n", opDuration)
			fmt.Fprintf(os.Stderr, "Total time: %v\n", loadDuration+parseDuration+opDuration)
		}

		if *jsonOutput {
			json.NewEncoder(os.Stdout).Encode(text)
		} else {
			fmt.Println(text)
		}

	case "chat":
		var input tokenizer.ChatTemplateInput

		if *chatFile != "" {
			// Read from file
			fileData, err := os.ReadFile(*chatFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading chat file: %v\n", err)
				os.Exit(1)
			}
			if err := json.Unmarshal(fileData, &input); err != nil {
				fmt.Fprintf(os.Stderr, "Error parsing chat file: %v\n", err)
				os.Exit(1)
			}
		} else if flag.NArg() > 0 {
			// Read from command line argument
			jsonStr := strings.Join(flag.Args(), " ")
			if err := json.Unmarshal([]byte(jsonStr), &input); err != nil {
				fmt.Fprintf(os.Stderr, "Error parsing chat JSON: %v\n", err)
				os.Exit(1)
			}
		} else {
			fmt.Fprintln(os.Stderr, "Error: chat mode requires either -chat-file or JSON input as argument")
			os.Exit(1)
		}

		startOp := time.Now()
		result, err := tok.ChatTemplate(input)
		opDuration := time.Since(startOp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error applying chat template: %v\n", err)
			os.Exit(1)
		}

		if *benchmark {
			fmt.Fprintf(os.Stderr, "Chat template time: %v\n", opDuration)
			fmt.Fprintf(os.Stderr, "Total time: %v\n", loadDuration+opDuration)
		}

		if *jsonOutput {
			json.NewEncoder(os.Stdout).Encode(result)
		} else {
			fmt.Print(result)
		}

	default:
		fmt.Fprintf(os.Stderr, "Error: invalid mode '%s'. Must be 'tokenize', 'detokenize', or 'chat'\n", *mode)
		os.Exit(1)
	}
}
