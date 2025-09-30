use std::fs;
use std::path::Path;
use std::io::Read;
use brotli::enc::BrotliEncoderParams;

fn main() {
    let tokenizers_dir = "tokenizers";

    // Create tokenizers directory if it doesn't exist
    fs::create_dir_all(tokenizers_dir).expect("Failed to create tokenizers directory");

    // Download and compress GLM-4.5 tokenizer files
    download_and_compress(
        "https://huggingface.co/zai-org/GLM-4.5/resolve/main/tokenizer.json",
        &format!("{}/glm-4.5-tokenizer.json", tokenizers_dir),
        &format!("{}/glm-4.5-tokenizer.json.br", tokenizers_dir)
    );

    download_and_compress(
        "https://huggingface.co/zai-org/GLM-4.5/resolve/main/tokenizer_config.json",
        &format!("{}/glm-4.5-tokenizer-config.json", tokenizers_dir),
        &format!("{}/glm-4.5-tokenizer-config.json.br", tokenizers_dir)
    );

    println!("cargo:rerun-if-changed=build.rs");
}

fn download_and_compress(url: &str, json_destination: &str, compressed_destination: &str) {
    let compressed_path = Path::new(compressed_destination);

    // Skip if compressed file already exists
    if compressed_path.exists() {
        println!("cargo:warning=Compressed file already exists: {}", compressed_destination);
        return;
    }

    println!("cargo:warning=Downloading {} to {}", url, json_destination);

    // Download the file
    let response = ureq::get(url)
        .call()
        .expect(&format!("Failed to download {}", url));

    let mut json_data = Vec::new();
    response.into_reader()
        .read_to_end(&mut json_data)
        .expect(&format!("Failed to read response from {}", url));

    // Save uncompressed version (for debugging/reference)
    let json_path = Path::new(json_destination);
    if !json_path.exists() {
        fs::write(json_path, &json_data)
            .expect(&format!("Failed to write {}", json_destination));
        println!("cargo:warning=Saved uncompressed file: {}", json_destination);
    }

    // Compress with Brotli (quality 11 for maximum compression)
    let mut compressed_data = Vec::new();
    let params = BrotliEncoderParams {
        quality: 11,
        lgwin: 22,
        ..Default::default()
    };

    brotli::BrotliCompress(
        &mut &json_data[..],
        &mut compressed_data,
        &params
    ).expect(&format!("Failed to compress {}", json_destination));

    // Save compressed version
    fs::write(compressed_path, &compressed_data)
        .expect(&format!("Failed to write {}", compressed_destination));

    let original_size = json_data.len();
    let compressed_size = compressed_data.len();
    let ratio = (compressed_size as f64 / original_size as f64) * 100.0;

    println!("cargo:warning=Compressed {} -> {} ({:.1}% of original)",
             json_destination, compressed_destination, ratio);
    println!("cargo:warning=Size: {} bytes -> {} bytes (saved {} bytes)",
             original_size, compressed_size, original_size - compressed_size);
}