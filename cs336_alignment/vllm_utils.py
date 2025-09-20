from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel
from vllm import LLM

def init_vllm(model_id, device, seed, gpu_memory_utilization):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )    

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def evaluate_vllm(vllm_model, reward_fn, prompts, sampling_params, ground_truth):
    outputs = vllm_model.generate(prompts, sampling_params)
    evaluation_data = []
    format_reward = 0
    answer_reward = 0
    total_reward = 0

    for output, ground_truth in zip(outputs, ground_truth):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, ground_truth)
        data = {
            "prompt": prompt,
            "generated_text": generated_text,
            "reward": reward,
        }
        evaluation_data.append(data)
        format_reward += reward["format_reward"]
        answer_reward += reward["answer_reward"]
        total_reward += reward["reward"]
    print(f"Number of prompts: {len(outputs)}")
    print(f"Format reward: {format_reward / len(outputs)}")
    print(f"Answer reward: {answer_reward / len(outputs)}")
    print(f"Reward: {total_reward / len(outputs)}")
    json.dump(evaluation_data, open("evaluation_epoch_2_data.json", "w"))