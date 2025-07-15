import tiktoken, inspect
enc = tiktoken.encoding_for_model("gpt-4")  # or "gpt-4o-mini"
print(enc.encode("hello world"))

from llmgine.settings import MAX_CONTEXT_TOKENS, PRUNE_TARGET_TOKENS
print(PRUNE_TARGET_TOKENS)