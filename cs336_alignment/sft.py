from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch

def forward(train_batch, model):
    input_ids = train_batch["input_ids"].to(model.device)
    labels = train_batch["labels"].to(model.device)
    logits = model(input_ids).logits    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_token_lengths = [len(ids) for ids in tokenizer(prompt_strs).input_ids]
    input_strs = [prompt + output for prompt, output in zip(prompt_strs, output_strs)]
    input_ids = tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True).input_ids
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    response_mask = torch.zeros(labels.shape, dtype=torch.bool)
    for i in range(len(prompt_token_lengths)):
        response_mask[i, prompt_token_lengths[i]:] = True
    
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def compute_entropy(logits):
    log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(torch.exp(log_p) * log_p, dim=-1)

def get_response_log_probs(model, input_ids, labels, return_token_entropy):
    logits = model(input_ids).logits
    selected_logits = logits.gather(dim=-1. index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs = selected_logits - torch.logsumexp(logits, dim=-1)
    ret = {"log_probs": log_probs}
    if return_token_entropy:
        entropy = compute_entropy(logits)
        ret["token_entropy"] = entropy
    return ret

def masked_normalize(tensor, mask, normalize_constant, dim):
    return torch.sum(tensor * (mask == 1).to(torch.int), dim=dim) / normalize_constant

def sft_microbatch_train_step(policy_log_probs, response_mask, gardient_accumulation_steps, normalize_constant=1.0):
    loss = masked_normalize(-policy_log_probs, response_mask, normalize_constant) / gardient_accumulation_steps
    loss.backward()
    return loss, {}

def log_generations(model, tokenizer):
    test_df = pd.read_parquet("/workspace/nano-grpo/data/competition_math/test.parquet")
    test_df = test_df.sample(5)
    r1_zero_prompt = open("cs336_alignment/prompts/r1_zero.prompt", "r").read()
    questions = test_df["problem"].tolist()
    ground_truth = test_df["solution"].tolist()
    prompt_strs = [r1_zero_prompt.format(question=question) for question in questions]
    for prompt in prompt_strs:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        # generate response for each prompt
        responses = model.generate(input_ids, max_new_tokens=1024, output_scores=True, return_dict_in_generate=True)
        print(responses)
    # get logits of the responses

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")  
    log_generations(model, tokenizer)  