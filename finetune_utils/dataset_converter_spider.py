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
    "[INST] {question} Given the schema:\n```sql\n{schema}\n```\nWhat SQL query would you write to address this? [/INST] \n```sql\n{query}\n```",
    "[INST] Examine the SQL query:\n```sql\n{query}\n```\nWhat question does this query aim to answer, assuming the following database schema? \n```sql\n{schema}\n``` [/INST] The query solves the question: {question}",
    "[INST] To tackle the problem '{question}', formulate an appropriate SQL query using the provided schema:\n```sql\n{schema}\n``` [/INST] \n```sql\n{query}\n```",
    "[INST] Given the SQL query:\n```sql\n{query}\n```\nAnd the database schema:\n```sql\n{schema}\n```\nInfer the question this query is designed to answer. [/INST] The question is: {question}",
    "[INST] Suppose you have a database with the following schema:\n```sql\n{schema}\n```\nHow would you retrieve the data needed to answer: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Consider the question: {question}\nUsing the schema provided:\n```sql\n{schema}\n```\nConstruct an SQL query to resolve this question. [/INST] \n```sql\n{query}\n```",
    "[INST] Analyze the given SQL query:\n```sql\n{query}\n```\nBased on the database schema:\n```sql\n{schema}\n```\nWhat question is this query intended to address? [/INST] The query answers the question: {question}",
    "[INST] Given the following database structure:\n```sql\n{schema}\n```\nHow would you write an SQL statement to find the information needed to resolve this inquiry: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] {question} Considering the tables and relationships defined below:\n```sql\n{schema}\n```\nWhat SQL command would you execute to handle this request? [/INST] \n```sql\n{query}\n```",
    "[INST] Examine the provided SQL code:\n```sql\n{query}\n```\nBased on the database layout:\n```sql\n{schema}\n```\nWhat problem does this SQL statement aim to solve? [/INST] The SQL code addresses the following issue: {question}",
    "[INST] To resolve '{question}', create a suitable SQL query using the given database blueprint:\n```sql\n{schema}\n``` [/INST] \n```sql\n{query}\n```", 
    "[INST] Analyze the SQL script:\n```sql\n{query}\n```\nAnd the database design:\n```sql\n{schema}\n```\nDeduce the challenge this script is designed to tackle. [/INST] The challenge is: {question}",
    "[INST] Imagine a database organized as follows:\n```sql\n{schema}\n```\nHow would you fetch the necessary data to respond to: {question} [/INST] \n```sql\n{query}\n```",
    "[INST] Ponder the dilemma: {question}\nUsing the database structure provided:\n```sql\n{schema}\n```\nDevise an SQL query to address this dilemma. [/INST] \n```sql\n{query}\n```",
    "[INST] Study the presented SQL command:\n```sql\n{query}\n```\nConsidering the database organization:\n```sql\n{schema}\n```\nWhat issue is this command intended to resolve? [/INST] The command solves the following problem: {question}"
]


def format_sample_with_template(sample):
    selected_template = random.choice(TEMPLATES)
    question = sample.get('question', '')
    raw_query = sample.get('query', '')
    schema_list = sample.get('schema', [])
    schema = "\n".join(schema_list)
    query = sqlparse.format(raw_query, reindent=True, keyword_case='upper')

    # Using exception handling to ignore placeholders if data is missing in the sample
    try:
        formatted_text = selected_template.format(question=question, query=query, schema=schema)
    except KeyError as e:
        # If some keys are missing, makeup or ignore and reformat
        formatted_text = selected_template.format(question=question, query=query)

    return formatted_text


def template_dataset(sample, tokenizer):
    formatted_text = format_sample_with_template(sample)
    sample['text'] = f"{tokenizer.bos_token}\n{formatted_text}\n{tokenizer.eos_token}"
    return sample