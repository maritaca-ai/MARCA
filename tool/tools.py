from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
import json
import os
from openai import AzureOpenAI


class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, input: str) -> str:
        pass

    def get_openai_compatible_schema(self):
        """Example of openai compatible schema:
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for provided coordinates in celsius.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"}
                        },
                        "required": ["latitude", "longitude"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        """
        parameters = self.get_parameters()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class SearchTool(BaseTool):
    def __init__(self):
        super().__init__(name="search", description="Search the web for information")

        self.url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }

    def run_search(self, query: str) -> str:

        payload = json.dumps(
            {"q": query, "location": "Brazil", "gl": "br", "hl": "pt-br"}
        )

        response = requests.request(
            "POST", self.url, headers=self.headers, data=payload
        )
        return response.json()

    def format_results(self, results: dict) -> str:
        organic_results = results.get("organic", [])
        result_str = ""
        for i, result in enumerate(organic_results):
            try:
                result_str += f"{i+1}. - Title: {result['title']} \n Link: {result['link']}\n Snippet: {result['snippet'] if 'snippet' in result else 'No snippet available'}\n\n"
            except:
                print(result)
                
                raise Exception(f"Error formatting result {i}: {result}")

        return result_str

    def execute(self, query: str) -> str:

        results = self.run_search(query)

        return self.format_results(results)

    def get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search the web for",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }


class PageScraperTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="page_scraper", description="Scrape a web page for information"
        )

        self.url = "https://scrape.serper.dev"
        self.headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }

    def run_scrape(self, url: str) -> str:

        payload = json.dumps({"url": url, "includeMarkdown": True})

        response = requests.request(
            "POST", self.url, headers=self.headers, data=payload
        )

        resposen_data = response.json()

        return resposen_data

    def format_results(self, results: dict) -> str:
        return results.get("markdown", "Error scraping page")

    def execute(self, url: str) -> str:

        results = self.run_scrape(url)

        return self.format_results(results)

    def get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The url to scrape"}
            },
            "required": ["url"],
            "additionalProperties": False,
        }


class MadeUpTool(BaseTool):
    def __init__(self, name: str, description: str, schema: dict):
        super().__init__(name=name, description=description)
        self.schema = schema

        self.model_name = "gpt-4o-2024-08-06"
        self.api_base = "https://maritacaopenai.openai.azure.com/"
        self.api_key = "YOUR_AZURE_API_KEY_HERE"
        self.api_version = "2024-08-01-preview"
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version,
        )

        assert type(self.schema) == dict, "Schema must be a dictionary"

    def execute(self, *args, **kwargs) -> str:

        args_str = ""
        kwargs_str = ""
        for arg in args:
            args_str += f"{arg}\n"
        for key, value in kwargs.items():
            kwargs_str += f"{key}: {value}\n"

        self.prompt = f"""
        You are an tool simulator.
        You will be given a schema of a function and a list of arguments.
        You will need to simulate the function execution with the given arguments. Returning an appropriate result.
        
        Schema:
        {self.schema}
        
        
        Answer only with the result of the function execution.
        
        Arguments:
        {args_str}
        {kwargs_str}
        """

        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": self.prompt}]
        )

        message = response.choices[0].message.content

        return message

    def get_openai_compatible_schema(self):
        return self.schema


class SubagentTool(BaseTool):
    """
    A tool that wraps a SubagentModel to allow the orchestrator to delegate queries.
    """
    def __init__(self, subagent_model):
        super().__init__(
            name="delegate_to_subagent",
            description="Delegate a query to a subagent that will search for information and provide an answer. Use this to break down complex queries into smaller, more manageable sub-queries."
        )
        self.subagent_model = subagent_model
        self.last_subagent_history = None  # Store last subagent history for debugging
        self.last_subagent_token_usage = None

    def execute(self, query: str) -> str:
        """
        Execute a query using the subagent model.
        Returns the subagent's response.
        The subagent's conversation history is stored in last_subagent_history for access by the orchestrator.
        """
        messages = [
            {"role": "user", "content": query}
        ]
        
        response, subagent_history, token_usage = self.subagent_model.generate_response(messages)
        
        # Store subagent history for debugging (can be accessed by orchestrator)
        self.last_subagent_history = {
            "query": query,
            "response": response,
            "conversation_history": subagent_history,
            "token_usage_details": token_usage,
        }
        self.last_subagent_token_usage = token_usage
        
        # Return the response
        return response

    def get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to delegate to the subagent. This should be a focused, specific question that the subagent can answer by searching the web.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }
