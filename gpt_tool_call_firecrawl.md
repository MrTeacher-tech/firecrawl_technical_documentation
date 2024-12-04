# Tool calling with GPT

*Learn to use OpenAI’s tool call feature with Firecrawl*

___

OpenAI’s tool calling allows LLMs to call functions that are defined in your code. This guide will show you how to utilise “tool calling” to call Firecrawl's crawl method. Along the way, we will:

- Install prerequisite packages and set up the environment
- Overview of how OpenAI’s tool calling feature works
- Use Firecrawl within an OpenAI tool call

Our final product will be a Python program that combines the OpenAI Chat API with Firecrawl's powerful web crawling functionality. The program will use Firecrawl to recursively crawl a URL, including its subpages, and gather content while bypassing common blockers. It will then process the scraped data into clean markdown or structured output, ideal for analysis by the OpenAI Chat API. This integration allows for intelligent, automated analysis of any website’s content.

___

## Setting Up

The first step to using openAI and/or Firecrawl within your python program is to install the packages using pip:

```system
pip install openai
```

```system
pip install firecrawl-py
```

Next, you will want to create a project directory

Create a directory for your project. In this directory, you should have your Python file (mine is called smart_crawl.py), and a .env file. It be structured like this:

```system
project/
├── smart_crawl.py
├── .env
```

Open your .env file and add your API keys:

```bash
FIRECRAWL_API_KEY = "YOUR_API_KEY"
OPENAI_API_KEY = "YOUR_API_KEY"
```

Now that our environment variables are set up, let's learn about OpenAI tool calling.

---

## What is OpenAI tool calling?


OpenAI LLMs can call a function you have defined in your code, this is called tool calling. To do this you first need to:

- Define the function (just as you would do with any function in Python)

- Define the function as a tool using the proper object format (shown below)

```python
TOOLS = [
    {
        "type": "function",  
        "function": {  
            "name": "your_function_name"    #name of the function you have defined (STR),
            "description": "desc_of_your_function"  #A description of your function, the LLM uses this to decide if the function should be called (STR)
            "parameters": {  
                "type": "object",
                "properties": {
                    "var_name": {   #name of the argument that the function takes (STR)
                        "type": "string",   #data type of passed argument (STR)
                        "description": "desc_of_argument"   #describe the arguement given to the function (STR)
                    },
                },
                "required": ["var_name"],  #Add any required arguments in a list (STR)
            }
        }
    },
    
]
```

> 🔔 **Note:** OpenAI expects a list of tools to be passed to the chat function. In the example above, we have a list with just one tool, but it’s important to note that you can include multiple tools in the list. Each tool in the list will follow the same structure.

Now that we've learned about tool calling, let's set up our Python file and get ready to define functions and tools.

___

## Loading Libraries and API Keys

Now that we've set up our environment and explored OpenAI tool calling, we can begin writing our Python code.

Let's start by adding the libraries we will require:

```python
import os
import json
from typing import Dict, Any        
from firecrawl import FirecrawlApp
from openai import OpenAI
from dotenv import load_dotenv
```
> 🔔 **Note:**
> **os:** used to load the api keys from environment variables.
>
> **json:** will read the tool description, that openAI saves as a string, and turn it into a Python dictionary.
> 
> **typing:** allows us to add expected data types as *type hints*. These have no effect on our code but give more context to the LLM when it reads our functions.
> 
> **dotenv** used to load enviornment variables from the .env file

Now that we have imported the necessary libraries, we will load our environment variables and initialize our clients.
```python
# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


# Initialize the FirecrawlApp and OpenAI client
firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
openai= OpenAI(api_key=openai_api_key)
```
___

## Defining Firecrawl Function and Setting up Tool List

As previously mentioned, we can pass defined functions to OpenAI models so that they can be used as tools when needed. This requires us to define a function and then define it as a tool for the model to use.

Let's write a simple function that utilizes Firecrawl's *crawl_url( )* method:

```python
def firecrawl_crawl(site: str) -> Dict[str, Any]:
    # Crawl a website:
    crawl_status = firecrawl.crawl_url(
    site, 
    params={
        'limit': 100, 
        'scrapeOptions': {'formats': ['markdown']}
    },
    poll_interval=30
    )

    return crawl_status
```

> 🔔 **Note:** The use of type hints is not required, but is reccomended as it allows the model to better understand the function

Now that our function is defined, let's define the function as a tool using the format the model expects:
```python
TOOLS = [
    {
        "type": "function", 
        "function": {  
            "name": "firecrawl_crawl",  #name of the function we just defined
            "description": "Recursively search through a urls subdomains, and gather the content. Begins with a specified URL, identifying links by looking at the sitemap and then crawling the website. Then converts collected data into clean markdown or structured output, perfect for LLM processing or any other task.",
            "parameters": { 
                "type": "object",
                "properties": {
                    "site": {   #name of argument given to firecrawl_crawl
                        "type": "string",   #type of argument
                        "description": "The url to crawl",
                    },
                },
                "required": ["site"],   #require the argument to run function
            }
        }
    },
    
]
```

Here, I have provided the name of the function I just defined as "name" and I have provided modified text from the Firecrawl documention for "description". The description is what the model is going to use to decide if it needs to use this tool. my description reads as:
>*"Recursively search through a urls subdomains, and gather the content. Begins with a specified URL, identifying links by looking at the sitemap and then crawling the website. Then converts collected data into clean markdown or structured output, perfect for LLM processing or any other task."*

Once we pass the tool list to the model, it will not automatically run any of our tool functions. Instead, it will return a list of the tools that it thinks it needs to better respond to our query. Thus, it is up to us to process these tool calls the model wants to make. We can write a function to accomplish this: 

```python
def process_tool_calls(tool_calls, messages):
    #tool_calls is a list of the functions the model wants to run
    #messages is our ongoing chat with the model

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        if function_name == "firecrawl_crawl":
            search_results = firecrawl_crawl(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "content": str(search_results),
                    "tool_call_id": tool_call.id,
                }
            )
            print(
                f"Context updated with: "
                f"firecrawl_crawl: ",
                function_args.get("site"),
            )
            
    return messages     #return the updated message context
```
With this complete, it is time to put it all together in a main function.
___

## Putting It All Together

Before we write our main function, let's cover what we have done so far. So far we've:

- installed the proper libraries onto our machine
- created a directory for our project
- added our API keys to a .env file
- loaded necessary libraries
- initialized clients with our API keys
- defined a function to call *crawl_url( )*
- set up tool objects for GPT model
- defined a function to process the tool calls the model will make

Looking forward, we are going to:

- tell the model its objective with a system primer
- ask the user for a site to crawl
- ask the user a question about the site
- initiate a chat with OpenAI
- process any tool calls the model makes
- if necessary, update chat context with tool results
- show results to user

Here is a main function that will put all that together:

```python
def main():
    
    #define the primer
    SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are an agent that has access to an advanced web scarpping tool, the tool provides you markdown text of a given site. Please provide the user with the information they are looking for by using the search tool provided.",
    }

    messages = [SYSTEM_MESSAGE]     #pass the primer to the chat
    
    while True:
        try:
            site = input("Enter a site to crawl: ")

            query = input("Enter a question:")

            #initial api call, this will decide if tools are needed
            messages.append({"role": "user", "content": query + " this site might be a helpful resource: " + site})
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
            )
            
            message = completion.choices[0].message
            tool_calls = message.tool_calls
            if tool_calls:  #if the model determines it needs to use a tool
                
                messages.append(message)
                messages = process_tool_calls(tool_calls, messages) #process the tool call
                messages.append(
                    {
                        "role": "user",
                        "content": "Answer my previous query based on the search results.",
                    }
                )
                completion = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                print("\n\n")
                print(completion.choices[0].message.content)
            else:
                print(message.content)
          
        except Exception as e:
            print(f"An error occurred: {str(e)}")
  
#run main on file execution
if __name__ == "__main__":
    main()

```
Now, let's run this and see the results!
___
## Running The Code

Here is a look at the full Python file:

```python
import os
import json
from typing import Dict, Any        

from firecrawl import FirecrawlApp
from openai import OpenAI

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


# Initialize the FirecrawlApp, OpenAI client, and Exa client
firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
openai= OpenAI(api_key=openai_api_key)

def firecrawl_crawl(site: str) -> Dict[str, Any]:
    # Crawl a website:
    crawl_status = firecrawl.crawl_url(
    site, 
    params={
        'limit': 100, 
        'scrapeOptions': {'formats': ['markdown']}
    },
    poll_interval=30
    )

    return crawl_status

TOOLS = [
    {
        "type": "function",  # Add this
        "function": {  # Wrap the existing definition in a 'function' key
            "name": "firecrawl_crawl",
            "description": "Recursively search through a urls subdomains, and gather the content. Begins with a specified URL, identifying links by looking at the sitemap and then crawling the website. Then converts collected data into clean markdown or structured output, perfect for LLM processing or any other task.",
            "parameters": {  # Note: changed from 'input_schema' to 'parameters'
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "description": "The url to crawl",
                    },
                },
                "required": ["site"],
            }
        }
    },
    
]


def process_tool_calls(tool_calls, messages):
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        if function_name == "firecrawl_crawl":
            search_results = firecrawl_crawl(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "content": str(search_results),
                    "tool_call_id": tool_call.id,
                }
            )
            print(
                f"Context updated with: "
                f"firecrawl_crawl: ",
                function_args.get("site"),
            )
    return messages



def main():
    
    #define the primer
    SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are an agent that has access to an advanced web scarpping tool, the tool provides you markdown text of a given site. Please provide the user with the information they are looking for by using the search tool provided.",
    }

    messages = [SYSTEM_MESSAGE]
    
    while True:
        try:
            site = input("Enter a site to crawl: ")

            query = input("Enter a question: ")

            #initial api call, this will decide if tools are needed
            messages.append({"role": "user", "content": query + " this site might be a helpful resource: " + site})
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
            )
            
            message = completion.choices[0].message
            tool_calls = message.tool_calls
            if tool_calls:  #if the model determines it needs to use a tool
                
                messages.append(message)
                messages = process_tool_calls(tool_calls, messages) #process the tool call
                messages.append(
                    {
                        "role": "user",
                        "content": "Answer my previous query based on the search results.",
                    }
                )
                completion = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                print("\n\n")
                print(completion.choices[0].message.content)
            else:
                print(message.content)
          
        except Exception as e:
            print(f"An error occurred: {str(e)}")
  

if __name__ == "__main__":
    main()

```
To run the script, enter the following command in your terminal (ensure you're in your project directory).

```system
python smart_crawl.py
```

That’s it, enjoy your smart crawl agent!
