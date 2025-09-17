from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pandas as pd
import json
r1_zero_prompt = open("cs336_alignment/prompts/r1_zero.prompt", "r").read()

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
    json.dump(evaluation_data, open("evaluation_data.json", "w"))


if __name__ == "__main__":
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model='Qwen/Qwen2.5-Math-1.5B')

    reward_fn = r1_zero_reward_fn
    test_df = pd.read_parquet("/workspace/nano-grpo/data/competition_math/test.parquet")
    questions = test_df["problem"].tolist()
    prompts = [r1_zero_prompt.format(question=question) for question in questions]
    ground_truth = test_df["solution"].tolist()
    evaluate_vllm(llm, reward_fn, prompts, sampling_params, ground_truth)