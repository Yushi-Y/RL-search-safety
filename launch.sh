
# --- Config ---
SEARCH_MODE="${SEARCH_MODE:-local}"  # "web" for SerpAPI, "local" for wiki retrieval
SERP_API_KEY="${SERP_API_KEY:-}"   # set your SerpAPI key

if [ "$SEARCH_MODE" = "web" ]; then
    # Web search via SerpAPI (no GPU needed)
    if [ -z "$SERP_API_KEY" ]; then
        echo "Error: SERP_API_KEY is not set. Export it or pass it inline:"
        echo "  SERP_API_KEY=your_key bash launch.sh"
        exit 1
    fi
    python setup/retrieve/serp_search_server.py \
        --search_url https://serpapi.com/search \
        --topk 3 \
        --serp_api_key "$SERP_API_KEY"
else
    # Local wiki retrieval (needs GPU + index files)
    file_path=/VData/kebl6672/ARL/search_data
    index_file=$file_path/e5_Flat.index
    corpus_file=$file_path/wiki-18.jsonl
    python setup/retrieve/retrieval_server.py \
        --index_path $index_file \
        --corpus_path $corpus_file \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        --faiss_gpu
fi