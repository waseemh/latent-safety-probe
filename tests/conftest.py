"""Shared fixtures: a tiny, randomly-initialised Qwen2-architecture model built locally
(no network), with a chat template, saved as safetensors."""

import pytest


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny_qwen2")

    # Minimal BPE tokenizer trained on a few sentences.
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"])
    corpus = ["how do i build a birdhouse at home", "how do i pick a lock on my door",
              "write a message to my neighbor", "ignore all previous instructions",
              "give me a recipe for pasta", "what time zone is lisbon in"] * 20
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
                                   eos_token="<|im_end|>")
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    fast.save_pretrained(d)

    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(d, safe_serialization=True)
    return str(d)
