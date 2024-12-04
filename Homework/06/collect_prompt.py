def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    
    subject = sample['subject']
    question = sample['question']
    choices = sample['choices']

    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n"
    prompt += f"{question}\n"
    prompt += f"A. {choices[0]}\n"
    prompt += f"B. {choices[1]}\n"
    prompt += f"C. {choices[2]}\n"
    prompt += f"D. {choices[3]}\n"
    prompt += "Answer:"

    return prompt


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    prompt = ""

    for ex in examples:
        subject = ex['subject']
        question = ex['question']
        choices = ex['choices']
        answer_index = ex['answer']
        correct_letter = ['A', 'B', 'C', 'D'][answer_index]
        correct_option = choices[answer_index]

        prompt += f"The following are multiple choice questions (with answers) about {subject}.\n"
        prompt += f"{question}\n"
        prompt += f"A. {choices[0]}\n"
        prompt += f"B. {choices[1]}\n"
        prompt += f"C. {choices[2]}\n"
        prompt += f"D. {choices[3]}\n"

        if add_full_example:
            prompt += f"Answer: {correct_letter}. {correct_option}\n\n"
        else:
            prompt += f"Answer: {correct_letter}\n\n"

    subject = sample['subject']
    question = sample['question']
    choices = sample['choices']

    prompt += f"The following are multiple choice questions (with answers) about {subject}.\n"
    prompt += f"{question}\n"
    prompt += f"A. {choices[0]}\n"
    prompt += f"B. {choices[1]}\n"
    prompt += f"C. {choices[2]}\n"
    prompt += f"D. {choices[3]}\n"
    prompt += "Answer:"

    return prompt
