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
  

if __name__ == "__main__":
    main()

