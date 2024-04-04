def format(sample):
    context = f"<s>[INST] Here's the dataset overview: {sample['context']}." if len(sample['context']) > 0 else None
    instruction = f" Provide SQL that answer: {sample['question']}"
    reponse = f" [/INST] {sample['answer']}"
    # join all parts together
    prompt = "".join([i for i in [context, instruction, reponse] if i is not None])
    return prompt

def template_dataset(sample, tokenizer):
    sample['text'] = f"{format(sample)}{tokenizer.eos_token}"
    return sample