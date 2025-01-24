from pathlib import Path
from pypdf import PdfReader
from litellm import completion
from openai import OpenAI
import json

client = OpenAI()
O1_MODEL = 'o1-mini'
GPT_MODEL = 'gpt-4o-mini'

o1_prompt = """
You are a legal analysis assistant. 
The first input you will receive will be a complex legal task that needs to be carefully reasoned through to solve.
Your task is to review the challenge, and create a detailed plan to solve this legal tasks and provide insights.
The document is already provided by the user and aware of the document. 
Start with the specific task/question you want the LLM to analyze rather than retrieving the full document.

You will have access to an LLM agent that is responsible for executing the plan that you create and will return results.


The LLM agent has access to the following functions:
    - query_legal_document(Question)
        - The function answers the queries regarding the legal documents.
        
When creating a plan for the LLM to execute, break your instructions into a logical, step-by-step order, using the specified format:
    - **Main actions are numbered** (e.g., 1, 2, 3).
    - **Sub-actions are lettered** under their relevant main actions (e.g., 1a, 1b).
        - **Sub-actions should start on new lines**
    - **Specify conditions using clear 'if...then...else' statements** (e.g., 'If the legal statement shows a profit, then...').
    - **For actions that require using one of the above functions defined**, write a step to call a function using backticks for the function name (e.g., `call the query_legal_document function`).
        - Ensure that the proper input arguments are given to the model for instruction. There should not be any ambiguity in the inputs.
    - **The last step** in the instructions should always be calling the `instructions_complete` function. This is necessary so we know the LLM has completed all of the instructions you have given it.
    - **Detailed steps** The plan generated must be extremely detailed and thorough with explanations at every step.
Use markdown format when generating the plan with each step and sub-step.

Please find the scenario below.
"""

gpt4o_system_prompt = """
You are a helpful assistant responsible for executing the policy on handling legal analysis tasks. 
Your task is to follow the policy exactly as it is written and perform the necessary actions.

You must explain your decision-making process across various steps.

# Steps

1. **Read and Understand Policy**: Carefully read and fully understand the given policy on handling legal analysis tasks.
2. **Identify the exact step in the policy**: Determine which step in the policy you are at, and execute the instructions according to the policy.
3. **Decision Making**: Briefly explain your actions and why you are performing them.
4. **Action Execution**: Perform the actions required by calling any relevant functions and input parameters.
5. **Use of Own Knowledge**: If specific industry trends or data are required and not available in the policy or context, use your own knowledge to provide insights.

POLICY:
{policy}
"""

TOOLS = [
{
        "type": "function",
        "function": {
            "name": "query_legal_document",
            "description": "Analyzes legal documents and provides answers to specific questions using AI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Question": {
                        "type": "string",
                        "description": "Query about the legal document content"
                    }
                },
                "required": ["Question"],
                "additionalProperties": False,
                "strict": True,
            },
        },
    },    
    {
        "type": "function",
        "function": {
            "name": "instructions_complete",
            "description": "Function should be called when we have completed ALL of the instructions.",
        },
    }
]


def query_legal_document(Question):
    global pdf_path_global
    # print("2", pdf_path)

    contexts = extract_text_from_pdf(pdf_path_global)
    # print(len(contexts))
    SYSTEM_PROMPT = """
    You are a specialized Legal Analysis AI focused on extracting concise, data-driven insights from legal documents. Provide short, precise answers with a brief explanation.

    **Remember**:  
    - Stick strictly to information from the documents.   
    - Avoid speculation or overly detailed elaboration.  
    - Provide a brief but clear thinking process, then a concise final answer.
    """

    USER_PROMPT = f"""
    Context:
    <Context>
        {contexts}
    </Context>

    Question: 
    <Question>
        {Question}
    </Question>

    Please provide a concise, reader-friendly answer.
    - Provide direct, concise answers suitable for FAQ format
    - Focus on accuracy and readability
    - Include only information found in the provided context
    - If information is incomplete or unclear, state this explicitly
    """

    response = completion(
        model="openrouter/google/gemini-flash-1.5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )
    print("Question:")
    print(Question)
    print("\nAnswer:")
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

function_mapping = {
    'query_legal_document': query_legal_document,
}
def extract_text_from_pdf(pdf_path):
    with Path(pdf_path).open("rb") as f:
        reader = PdfReader(f)
        text = "\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def call_o1(scenario):
    prompt = f"""
    {o1_prompt}
    
    Scenario:
    {scenario}
    
    Please provide the next steps in your plan.
    """

    full_response = ""
    response = client.chat.completions.create(
        model=O1_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end='')
            full_response += chunk.choices[0].delta.content
            
    return full_response  

def call_gpt4o(message_list, plan):
    gpt4o_policy_prompt = gpt4o_system_prompt.replace("{policy}", plan)
    messages = [
        {'role': 'system', 'content': gpt4o_policy_prompt},
    ]
    while True:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            tools=TOOLS,
            parallel_tool_calls=False
        )
        
        assistant_message = response.choices[0].message.to_dict()
        # print(assistant_message)
        messages.append(assistant_message)

        if 'tool_calls' not in assistant_message:
            append_message(message_list, {'type': 'assistant', 'content': assistant_message.get('content', '')})

        if (response.choices[0].message.tool_calls and
            response.choices[0].message.tool_calls[0].function.name == 'instructions_complete'):
            break

        if not response.choices[0].message.tool_calls:
            continue

        for tool in response.choices[0].message.tool_calls:
            tool_id = tool.id
            function_name = tool.function.name
            input_arguments_str = tool.function.arguments
            
            append_message(message_list, {'type': 'tool_call', 'function_name': function_name, 'arguments': input_arguments_str})

            try:
                input_arguments = json.loads(input_arguments_str)
                    
            except (ValueError, json.JSONDecodeError):
                continue

            if function_name in function_mapping:
                print("## Function Call", function_name)
                try:
                    function_response = function_mapping[function_name](**input_arguments)
                except Exception as e:
                    function_response = {'error': str(e)}
            else:
                function_response = {'error': f"Function '{function_name}' not implemented."}

            try:
                serialized_output = json.dumps(function_response)
            except (TypeError, ValueError):
                serialized_output = str(function_response)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": serialized_output
            })

            append_message(message_list, {'type': 'tool_response', 'function_name': function_name, 'response': serialized_output})

    return messages

def append_message(message_list, message):
    message_list.append(message)
    # Optionally, print the message for immediate feedback
    message_type = message.get('type', '')
    if message_type == 'status':
        print(message['message'])
    elif message_type == 'plan':
        print("\nPlan:\n", message['content'])
    elif message_type == 'assistant':
        print("\nAssistant:\n", message['content'])
    elif message_type == 'function_call':
        print(f"\nFunction call: {message['function_name']} with arguments {message['arguments']}")
    elif message_type == 'function_response':
        print(f"\nFunction response for {message['function_name']}: {message['response']}")
    else:
        # Handle any other message types or default case
        print(message.get('content', ''))

def process_scenario(message_list, scenario):
    append_message(message_list, {'type': 'status', 'message': 'Generating plan...'})

    plan = call_o1(scenario)

    # append_message(message_list, {'type': 'plan', 'content': plan})

    append_message(message_list, {'type': 'status', 'message': 'Executing plan...'})

    messages = call_gpt4o(message_list, plan)

    append_message(message_list, {'type': 'status', 'message': 'Processing complete.'})

    return messages

def get_final_report(messages, scenario):
    system_prompt = """
    You are a legal analysis assistant. 
    Your task is to generate a final report based on the provided messages.
    The final report should be in markdown format.
    """
    user_prompt = f"""
    Using the messages below, generate a final report in markdown format.

    <message>
        {messages}
    </message>

    <Research>
        {scenario}
    </Research>

    Provide only the final report of this case.
    Avoid preamble statements.
    """
    response = completion(
        model="openrouter/minimax/minimax-01",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    )               
    return response.choices[0].message.content

def clean_markdown(content):
    return content.replace('```markdown', '').replace('```', '')

def process_legal_document(pdf_path: Path, custom_scenario: str) -> str:
    global pdf_path_global
    pdf_path_global = pdf_path
    
    if custom_scenario != None and custom_scenario.strip() != "":
        scenario = custom_scenario
    else:
        scenario = """
        Provide a detailed summary of this legal case:

        Case details (name, court, judges, date of the judgment)
        Facts of the case [Parties involved , Background and Legal Actions]
        Key Issues [in numbered lists]
        Court's Reasonings [for each issues in numbered lists with heading and reasonings in one detailed paragraph] 
        Final Decision [in numbered lists]

        Please be specific and focus on key information.
        """
    print(scenario)
    messages = process_scenario([], scenario)
    final_report = get_final_report(messages, scenario)
    return clean_markdown(final_report)

