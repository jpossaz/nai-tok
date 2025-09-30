"""
Generate test cases for apply_chat_template with various configurations.
Saves input as JSON and rendered output as TXT for Rust comparison.
Uses the zai-org/GLM-4.5 tokenizer from Hugging Face.
"""

import json
import os
from pathlib import Path
from transformers import AutoTokenizer

# Load the GLM-4.5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5", trust_remote_code=True)

# Create output directories
output_dir = Path("cases")
output_dir.mkdir(exist_ok=True)


def save_test_case(name: str, messages: list, config: dict, output: str):
    """Save test case as JSON input and TXT output"""
    # Combine messages and config into input
    input_data = {
        "messages": messages,
        "config": config,
    }

    # Save JSON input
    json_path = output_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(input_data, f, indent=2)

    # Save TXT output
    txt_path = output_dir / f"{name}.txt"
    with open(txt_path, "w") as f:
        f.write(output)

    print(f"✓ Saved {name}")


def generate_test_cases():
    """Generate the most important test cases"""

    # Test 1: Basic conversation with generation prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("01_basic_with_gen_prompt", messages, config, output)

    # Test 2: Thinking enabled
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Solve this: 2 + 2"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("02_thinking_enabled", messages, config, output)

    # Test 3: Thinking disabled (/nothink)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Solve this: 2 + 2"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("03_thinking_disabled", messages, config, output)

    # Test 4: Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are an expert programmer."},
        {"role": "user", "content": "How do I read a file in Python?"},
        {"role": "assistant", "content": "Use `open()` with a context manager:\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"},
        {"role": "user", "content": "What about writing?"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("04_multi_turn", messages, config, output)

    # Test 5: No system prompt
    messages = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have real-time data."},
        {"role": "user", "content": "Make a guess?"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("05_no_system_prompt", messages, config, output)

    # Test 6: Assistant with reasoning_content (thinking disabled)
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 15 * 24?"},
        {
            "role": "assistant",
            "reasoning_content": "15 * 24 = 15 * (20 + 4) = 300 + 60 = 360",
            "content": "The answer is 360."
        },
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": False,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("06_reasoning_content_think_disabled", messages, config, output)

    # Test 7: Assistant with reasoning_content (thinking enabled)
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 15 * 24?"},
        {
            "role": "assistant",
            "reasoning_content": "15 * 24 = 15 * (20 + 4) = 300 + 60 = 360",
            "content": "The answer is 360."
        },
    ]
    config = {
        "add_generation_prompt": False,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("07_reasoning_content_think_enabled", messages, config, output)

    # Test 8: Multi-turn with reasoning_content
    messages = [
        {"role": "user", "content": "Is 17 prime?"},
        {
            "role": "assistant",
            "reasoning_content": "Check divisibility up to sqrt(17) ≈ 4.12. Not divisible by 2, 3. So 17 is prime.",
            "content": "Yes, 17 is prime."
        },
        {"role": "user", "content": "What about 18?"},
    ]
    config = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": True,
    }
    output = tokenizer.apply_chat_template(messages, **config)
    save_test_case("08_multi_turn_reasoning", messages, config, output)


if __name__ == "__main__":
    print("Generating test cases for GLM-4.5 tokenizer...\n")
    generate_test_cases()
    print(f"\n✅ All test cases generated in {output_dir}/")
    print("Each test case has:")
    print("  - .json file with messages and config")
    print("  - .txt file with rendered output")