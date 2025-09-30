import init, { tokenize, detokenize, decode_token, vocab_size } from './pkg/nai_tokenizers_web.js';

let initialized = false;
let initPromise = null;

/**
 * Initialize the WASM module
 * @returns {Promise<void>}
 */
async function ensureInitialized() {
    if (initialized) return;

    if (!initPromise) {
        initPromise = init().then(() => {
            initialized = true;
            console.log('WASM tokenizer initialized');
        });
    }

    return initPromise;
}

self.onmessage = async function(e) {
    const { type, data, id } = e.data;

    try {
        await ensureInitialized();

        switch (type) {
            case 'tokenize': {
                const { text, keepSpecialTokens = true } = data;
                const ids = tokenize(text, keepSpecialTokens);

                // Decode each token to get its text
                const tokens = [];
                for (let i = 0; i < ids.length; i++) {
                    const token_text = decode_token(ids[i], keepSpecialTokens);
                    tokens.push({
                        id: ids[i],
                        text: token_text
                    });
                }

                self.postMessage({
                    type: 'tokenize_result',
                    data: { tokens, ids },
                    id
                });
                break;
            }

            case 'detokenize': {
                const { ids, keepSpecialTokens = true } = data;
                const result = detokenize(ids, keepSpecialTokens);
                self.postMessage({
                    type: 'detokenize_result',
                    data: { text: result },
                    id
                });
                break;
            }

            case 'get_info': {
                const vocabSize = vocab_size();
                self.postMessage({
                    type: 'info_result',
                    data: { vocabSize },
                    id
                });
                break;
            }

            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        console.error('Worker error:', error);
        self.postMessage({
            type: 'error',
            error: error.message || String(error),
            id
        });
    }
};

self.onerror = (error) => {
    console.error('Worker script error:', error);
    self.postMessage({
        type: 'error',
        error: error.message || String(error)
    });
};