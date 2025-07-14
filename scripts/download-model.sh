#!/bin/bash

# Download script for GGUF models
# Supports Phi-3-mini-4k-instruct, Gemma-2B-IT, and TinyLlama

# --- Logging Configuration ---
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/download_model_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages to both console and file
log() {
    echo "$1" | tee -a "$LOG_FILE"
}
# --------------------------

log "🚀 Starting model download process at $(date)..."
log ""

# --- Model Selection ---
# Pass 'phi3', 'gemma', or 'tinyllama' as arguments. Defaults to downloading all three.
if [ "$#" -eq 0 ]; then
    MODEL_CHOICES=("phi3" "gemma" "tinyllama")
else
    MODEL_CHOICES=("$@")
fi
MODEL_DIR="./models"

# --- Main Loop ---
for MODEL_CHOICE in "${MODEL_CHOICES[@]}"; do
    log "--- Processing model: $MODEL_CHOICE ---"

    # --- Model Configurations ---
    MODEL_NAME=""
    MODEL_URL=""
    MODEL_FILE=""
    REPO_ID=""
    REQUIRES_AUTH=false

    if [ "$MODEL_CHOICE" = "phi3" ]; then
        MODEL_NAME="Phi-3-mini-4k-instruct"
        MODEL_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
        MODEL_FILE="phi-3-mini-4k-instruct-q4.gguf"
        REPO_ID="microsoft/Phi-3-mini-4k-instruct-gguf"
    elif [ "$MODEL_CHOICE" = "gemma" ]; then
        MODEL_NAME="Gemma-2B-IT (Q4_K_M)"
        MODEL_URL="https://huggingface.co/TheBloke/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf"
        MODEL_FILE="gemma-2b-it.Q4_K_M.gguf"
        REPO_ID="TheBloke/gemma-2b-it-GGUF"
        REQUIRES_AUTH=true
    elif [ "$MODEL_CHOICE" = "tinyllama" ]; then
        MODEL_NAME="TinyLlama-1.1B-Chat-v1.0 (Q4_K_M)"
        MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        REPO_ID="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        REQUIRES_AUTH=false
    else
        log "❌ Warning: Invalid model choice '$MODEL_CHOICE'. Skipping."
        log "Please use 'phi3', 'gemma', or 'tinyllama'."
        log ""
        continue
    fi
    # --------------------------

    log "📖 Selected Model: $MODEL_NAME"
    log "📁 Target directory: $MODEL_DIR"
    log ""

    # Create models directory if it doesn't exist
    if [ ! -d "$MODEL_DIR" ]; then
        log "📁 Creating models directory..."
        mkdir -p "$MODEL_DIR"
    fi

    # Check if model already exists
    if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
        log "✅ Model already exists at $MODEL_DIR/$MODEL_FILE. Skipping download."
        log ""
        continue
    fi

    # --- Download Logic ---
    DOWNLOAD_SUCCESS=false

    # Prefer huggingface-cli for authenticated downloads if available
    if [ "$REQUIRES_AUTH" = true ] && command -v huggingface-cli &> /dev/null; then
        log "✅ 'huggingface-cli' found. Using it for a more reliable download."
        log "   ================================================================================"
        log "   IMPORTANT: Ensure you have run 'huggingface-cli login' or that the"
        log "              HUGGING_FACE_HUB_TOKEN environment variable is correctly set."
        log "   You MUST also accept the model's license at: https://huggingface.co/google/gemma-2b-it"
        log "   ================================================================================"
        
        huggingface-cli download "$REPO_ID" "$MODEL_FILE" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False --repo-type model 2>&1 | tee -a "$LOG_FILE"
        if [ "${PIPESTATUS[0]}" -eq 0 ]; then
            log "✅ Download via huggingface-cli successful."
            DOWNLOAD_SUCCESS=true
        else
            log "⚠️ 'huggingface-cli' download failed. Check log for details. Will attempt fallback to curl/wget."
        fi
    fi

    # Fallback to curl/wget if huggingface-cli is not used or fails
    if [ "$DOWNLOAD_SUCCESS" = false ]; then
        # Notify user if falling back on an auth-required download
        if [ "$REQUIRES_AUTH" = true ] && ! command -v huggingface-cli &> /dev/null; then
            log "⚠️ 'huggingface-cli' not found. Falling back to curl/wget."
            log "   For a more reliable download, please install it: pip install -U huggingface_hub"
            log "   Then run: huggingface-cli login"
        fi

        AUTH_ARGS_WGET=()
        AUTH_ARGS_CURL=()
        if [ "$REQUIRES_AUTH" = true ]; then
            if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
                log "❌ FATAL: The model '$MODEL_NAME' requires authentication."
                log "           HUGGING_FACE_HUB_TOKEN is not set. Cannot continue with fallback."
                continue
            fi
            AUTH_ARGS_WGET=("--header=Authorization: Bearer $HUGGING_FACE_HUB_TOKEN")
            AUTH_ARGS_CURL=("-H" "Authorization: Bearer $HUGGING_FACE_HUB_TOKEN")
        fi

        log "⬇️  Downloading $MODEL_NAME via curl/wget..."
        log "🔗 from $MODEL_URL"
        
        if command -v wget &> /dev/null; then
            wget "${AUTH_ARGS_WGET[@]}" -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL" --progress=bar:force 2>&1 | tee -a "$LOG_FILE"
            if [ "${PIPESTATUS[0]}" -eq 0 ]; then
                DOWNLOAD_SUCCESS=true
            fi
        elif command -v curl &> /dev/null; then
            curl -L -# "${AUTH_ARGS_CURL[@]}" -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL" 2>&1 | tee -a "$LOG_FILE"
            if [ "${PIPESTATUS[0]}" -eq 0 ]; then
                DOWNLOAD_SUCCESS=true
            fi
        else
            log "❌ Error: Neither wget nor curl is available. Cannot download this model."
        fi
    fi

    # Verify download
    if [ "$DOWNLOAD_SUCCESS" = true ] && [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
        log "✅ Model '$MODEL_NAME' downloaded successfully!"
        FILE_SIZE=$(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1)
        log "📊 File size: $FILE_SIZE"
        log "📝 Model path: $MODEL_DIR/$MODEL_FILE"
    else
        log "❌ Error: Download failed for $MODEL_NAME. Check $LOG_FILE for details."
        # Clean up partially downloaded file if it exists
        [ -f "$MODEL_DIR/$MODEL_FILE" ] && rm "$MODEL_DIR/$MODEL_FILE"
    fi
    log "" # Add a blank line for readability between models
done

log "---"
log "🎉 All model processing complete."
log ""
log "🎯 Next steps:"
log "1. Set the MODEL_PATH environment variable to the desired model file in the '$MODEL_DIR/' directory."
log "2. Start the server: python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000"
log ""
log "📋 Process log saved to: ${LOG_FILE}"
