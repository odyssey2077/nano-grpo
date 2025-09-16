from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pandas as pd

prompts = [
"Hello, my name is",
"The president of the United States is",
"The capital of France is",
"The future of AI is",
]


def evaluate_vllm(vllm_model, reward_fn, prompts, sampling_params, ground_truth):
    outputs = vllm_model.generate(prompts, sampling_params)
    evaluation_data = []
    format_reward = 0
    answer_reward = 0
    reward = 0

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
        reward += reward["reward"]
    print(f"Number of prompts: {len(outputs)}")
    print(f"Format reward: {format_reward / len(outputs)}")
    print(f"Answer reward: {answer_reward / len(outputs)}")
    print(f"Reward: {reward / len(outputs)}")


if __name__ == "__main__":
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model='Qwen/Qwen2.5-Math-1.5B')

    reward_fn = r1_zero_reward_fn
    test_df = pd.read_parquet("data/competition_math/test.parquet")
    prompts = test_df["problem"].tolist()
    ground_truth = test_df["solution"].tolist()
    evaluate_vllm(llm, reward_fn, prompts, sampling_params, ground_truth)