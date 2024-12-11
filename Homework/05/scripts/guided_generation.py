import torch

from scripts.compute_reward import compute_reward


def generate_with_reward_guidance(
        main_model, main_tokenizer,
        reward_model, reward_tokenizer,
        N=16,
        device='cpu',
    ):
    """
    Generate text samples using a main model and select the best sample based on a reward model's guidance.

    This function generates multiple text samples from a main model, evaluates each sample using a reward model,
    and returns the sample with the highest reward score. The process is guided by the reward model to select
    the most desirable output.

    Parameters:
    main_model: The language model used to generate text samples.
    main_tokenizer: The tokenizer for main_model
    reward_model: The model used to compute reward scores for the generated samples.
    reward_tokenizer: The tokenizer for reward_model
    N (int, optional): The number of text samples to generate. Default is 16.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    str: The generated text sample with the highest reward score.
    """

    prompt = ""
    prompt_enc = main_tokenizer([prompt], return_tensors='pt')
    input_ids = prompt_enc['input_ids'].to(device)
    attention_mask = prompt_enc['attention_mask'].to(device)

    samples = []
    for _ in range(N):
        gen = main_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=True
        )
        text = main_tokenizer.decode([gen[0].item()])
        samples.append(text)

    rewards = compute_reward(reward_model, reward_tokenizer, samples)
    best_idx = torch.argmax(rewards).item()
    return samples[best_idx]
