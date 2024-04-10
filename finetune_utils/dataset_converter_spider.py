import random
import sqlparse


TEMPLATES = [
    "[INST] How would you write SQL to retrieve data: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] How would you write SQL to answer: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] How can the question '{question}' be answered using SQL [/INST] \n```sql\n{query}\n```",
    "[INST] Provide the SQL query that solves this problem: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Create a SQL query to find the answer to: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] What SQL would you use to address the question: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Please formulate a SQL statement to explore: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Generate an SQL command to solve for: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Craft a SQL query that answers the following: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Draft an SQL query to get information regarding: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Write SQL code to fetch data based on this query: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Develop a SQL statement that would answer the query: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Describe the SQL needed to handle the question: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] SQL coding: How would you handle the query '{question}'? [/INST] \n```sql\n{query}\n```",
    "[INST] Write an SQL statement to extract data based on the following requirement: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] How do you formulate an SQL query to address the problem described here: '{question}'? [/INST] \n```sql\n{query}\n```",
    "[INST] Given the database task, '{question}', construct the appropriate SQL command.[/INST] \n```sql\n{query}\n```",
    "[INST] Construct a SQL script to fetch data as per the question: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] For the challenge outlined in '{question}', what SQL operation would you perform? [/INST] \n```sql\n{query}\n```"
]


def format_sample_with_template(sample):
    selected_template = random.choice(TEMPLATES)
    question = sample.get('question', '')
    raw_query = sample.get('query', '')
    query = sqlparse.format(raw_query, reindent=True, keyword_case='upper')

    # Using exception handling to ignore placeholders if data is missing in the sample
    try:
        formatted_text = selected_template.format(question=question, query=query)
    except KeyError as e:
        # If some keys are missing, makeup or ignore and reformat
        formatted_text = selected_template.format(question=question, query=query)

    return formatted_text


def template_dataset(sample, tokenizer):
    formatted_text = format_sample_with_template(sample)
    sample['text'] = f"{tokenizer.bos_token}\n{formatted_text}\n{tokenizer.eos_token}"
    return sample