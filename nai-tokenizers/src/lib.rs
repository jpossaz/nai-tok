#[cfg(feature = "glm45")]
pub mod glm45_tokenizer {
    use anyhow::Result;

    #[cfg(feature = "native")]
    use tokenizers::Tokenizer;

    #[cfg(feature = "wasm")]
    use tokenizers_wasm::Tokenizer;

    pub fn load() -> Result<Tokenizer> {
        // Load compressed tokenizer data
        let compressed_data = include_bytes!("../tokenizers/glm-4.5-tokenizer.json.br");

        // Decompress with Brotli
        let mut decompressed_data = Vec::new();
        brotli::BrotliDecompress(
            &mut &compressed_data[..],
            &mut decompressed_data
        ).map_err(|e| anyhow::anyhow!("Failed to decompress tokenizer: {}", e))?;

        // Parse from decompressed bytes
        let tokenizer = Tokenizer::from_bytes(&decompressed_data)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(tokenizer)
    }

    lazy_static::lazy_static! {
        pub static ref GLM45_TOKENIZER: Tokenizer = load().expect("Failed to load GLM-4.5 tokenizer");
    }

    #[derive(Clone, Copy)]
    pub enum SpecialTokens {
        Ignore,
        Keep,
    }

    impl From<SpecialTokens> for bool {
        fn from(val: SpecialTokens) -> bool {
            match val {
                SpecialTokens::Ignore => false,
                SpecialTokens::Keep => true,
            }
        }
    }

    pub fn tokenize(input: &str, special_tokens: SpecialTokens) -> Result<Vec<u32>> {
        let encoding = GLM45_TOKENIZER
            .encode(input, special_tokens.into())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn detokenize(ids: &[u32], special_tokens: SpecialTokens) -> Result<String> {
        let special_tokens: bool = special_tokens.into();
        let decoded = GLM45_TOKENIZER
            .decode(ids, !special_tokens)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(decoded)
    }

    #[cfg(test)]
    mod tests {
        use std::vec;

        use super::*;
        #[test]
        fn test_tokenize() {
            let input = "[gMASK]this is a test where da goose is cooked<|system|>no<|user|>yes<|assistant|>maybe";
            let expected_output = vec![
                151331, 574, 374, 264, 1273, 1380, 2994, 61701, 374, 28998, 151335, 2152, 151336,
                9689, 151337, 36569,
            ];
            let output = tokenize(input, SpecialTokens::Keep).unwrap();
            assert_eq!(output, expected_output);
        }

        #[test]
        fn test_detokenize() {
            let input = vec![
                151331, 574, 374, 264, 1273, 1380, 2994, 61701, 374, 28998, 151335, 2152, 151336,
                9689, 151337, 36569,
            ];
            let expected_output = "[gMASK]this is a test where da goose is cooked<|system|>no<|user|>yes<|assistant|>maybe";
            let output = detokenize(&input, SpecialTokens::Keep).unwrap();
            assert_eq!(output, expected_output);
        }
    }
}
