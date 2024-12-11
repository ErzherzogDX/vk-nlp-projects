from tqdm.auto import tqdm

from scripts.compute_reward import compute_reward


def eval_reward_model(reward_model, reward_tokenizer, test_dataset, target_label, device='cpu'):
    """
    Evaluate the performance of a reward model by comparing reward scores for chosen and rejected reviews. 

    This function selects reviews from a test dataset based on a target label and evaluates the reward model's
    ability to assign higher scores to chosen reviews compared to rejected ones. The evaluation is performed
    in batches for efficiency.
    Note that reward scores are compared on corresponding chosen and rejected reviews: 
        chosen_reviews[0] vs rejected_reviews[0], 
        chosen_reviews[1] vs rejected_reviews[1],
        etc.

    Parameters:
    reward_model: The model used to compute the reward scores
    reward_tokenizer: The tokenizer for reward_model
    tes_dataset: test Dataset
    target_label (0 or 1): The label used to select chosen reviews. Reviews with this label are considered chosen,
                  while others are considered rejected.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    float: The accuracy of the reward model, calculated as the proportion of times the model assigns a higher
           reward score to the chosen review compared to the rejected review.

    Example:
    >>> accuracy = eval_reward_model(my_reward_model, my_reward_tokenizer, test_data, target_label=1)
    >>> print(f"Model accuracy: {accuracy:.2%}")
    """

    chosen_texts = [x['text'] for x in test_dataset if x['label'] == target_label]
    rejected_texts = [x['text'] for x in test_dataset if x['label'] != target_label]

    size = min(len(chosen_texts), len(rejected_texts))
    chosen_texts = chosen_texts[:size]
    rejected_texts = rejected_texts[:size]

    assert len(chosen_texts) == len(rejected_texts)

    batch_size = 64
    correct = 0
    total = 0
    for i in range(0, size, batch_size):
        batch_chosen = chosen_texts[i:i + batch_size]
        batch_rejected = rejected_texts[i:i + batch_size]

        chosen_rewards = compute_reward(reward_model, reward_tokenizer, batch_chosen)
        rejected_rewards = compute_reward(reward_model, reward_tokenizer, batch_rejected)

        correct += (chosen_rewards > rejected_rewards).sum().item()
        total += len(batch_chosen)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy