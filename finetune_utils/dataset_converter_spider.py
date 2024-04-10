import random

# Define multiple templates for formatting the sample without context
def template_a(sample):
    prefix = "<>[INST]"
    instruction = f"How would you write SQL to answer: {sample['question']}"
    sql_markdown = f"```sql\n{sample['query']}\n```"
    return f"{prefix} {instruction} [/INST] \n{sql_markdown}"

def template_b(sample):
    prefix = "<s>[INST]"
    instruction = f"How can the question '{sample['question']}' be answered using SQL"
    sql_markdown = f"```sql\n{sample['query']}\n```"
    return f"{prefix} {instruction} [/INST] \n{sql_markdown}"

def template_c(sample):
    prefix = "<s>[INST]"
    instruction = f"Please demonstrate how SQL can be used to address the question: '{sample['question']}'."
    sql_markdown = f"```sql\n{sample['query']}\n```"
    return f"{prefix} {instruction} [/INST] \n{sql_markdown}"

def template_d(sample):
    prefix = "<s>[INST]"
    instruction = f"Let's collaborate on writing an SQL query to solve for '{sample['question']}'. How should we approach it?"
    sql_markdown = f"```sql\n{sample['query']}\n```"
    return f"{prefix} {instruction} [/INST] \n{sql_markdown}"

def template_e(sample):
    prefix = "<s>[INST]"
    instruction = f"Could you formulate an SQL query to address the inquiry: '{sample['question']}'"
    sql_markdown = f"```sql\n{sample['query']}\n```"
    return f"{prefix} {instruction} [/INST] \n{sql_markdown}"

def template_f(sample):
    prefix = "<s>[INST]"
    instruction = f"You've been tasked with finding information related to :'{sample['question']}'"
    sql_markdown = f"```sql\n{sample['query']}\n```"
    return f"{prefix} {instruction} [/INST] \n{sql_markdown}"

# Function to randomly select a template and format the sample
def format_sample_randomly(sample):
    templates = [template_a, template_b, template_c, template_d, template_e, template_f]
    chosen_template = random.choice(templates)
    return chosen_template(sample)

# Modified template_dataset function that uses the new random format function
def template_dataset(sample, tokenizer):
    sample['text'] = f"{format_sample_randomly(sample)}\n{tokenizer.eos_token}"
    return sample
