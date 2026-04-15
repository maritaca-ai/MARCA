from abc import ABC, abstractmethod
import json
from typing import List, Dict, Any, Optional
import openai
from collections import defaultdict
from tool.tools import BaseTool


class BaseModel(ABC):
    def __init__(self, model_name: str, model_type: str, api_base: str, api_key: str):
        self.model_name = model_name
        self.model_type = model_type  # 'assistant' or 'user'
        self.api_base = api_base
        self.api_key = api_key

        self.token_usage = {"total": defaultdict(int), "last_call": defaultdict(int)}

        if "azure" in api_base:
            api_version = "2024-08-01-preview"
            print(api_base)

            self.client = openai.AzureOpenAI(
                api_key=api_key, azure_endpoint=api_base, api_version=api_version
            )
        else:
            self.client = openai.OpenAI(api_key=api_key, base_url=api_base)

    def update_token_usage(self, response: openai.ChatCompletion):
        self.token_usage["total"]["prompt_tokens"] += response.usage.prompt_tokens
        self.token_usage["total"][
            "completion_tokens"
        ] += response.usage.completion_tokens
        self.token_usage["total"]["total_tokens"] += response.usage.total_tokens
        self.token_usage["last_call"]["prompt_tokens"] = response.usage.prompt_tokens
        self.token_usage["last_call"][
            "completion_tokens"
        ] = response.usage.completion_tokens
        self.token_usage["last_call"]["total_tokens"] = response.usage.total_tokens

    def expand_usage(self, response: openai.ChatCompletion):
        token_details = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }

        completion_details = getattr(response.usage, "completion_tokens_details", None)
        if completion_details and getattr(completion_details, "reasoning_tokens", None):
            token_details["reasoning_tokens"] = completion_details.reasoning_tokens

        prompt_details = getattr(response.usage, "prompt_tokens_details", None)
        if prompt_details and getattr(prompt_details, "cached_tokens", None):
            token_details["cached_tokens"] = prompt_details.cached_tokens

        return token_details

    def get_token_usage(self):
        return self.token_usage

    @abstractmethod
    def generate_response(
        self, conversation_history: List[Dict[str, Any]], **kwargs
    ) -> str:
        """
        Generate a response given the conversation history.
        To be implemented by subclasses.
        """
        pass


class AssistantModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        super().__init__(model_name, "assistant", api_base, api_key)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def generate_response(self, conversation_history: List[Dict[str, Any]], **kwargs):
        # Placeholder for OpenAI-compatible API call

        if self.system_prompt:

            messages = [
                {"role": "system", "content": self.system_prompt},
                *conversation_history,
            ]
        else:
            messages = conversation_history

        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        if self.reasoning_effort:
            request_kwargs["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**request_kwargs)

        self.update_token_usage(response)

        return response.choices[0].message.content, None


class AssistantWithTools(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        tools: List[BaseTool] = [],
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        super().__init__(model_name, "assistant", api_base, api_key)
        self.system_prompt = system_prompt
        self.tools = tools
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

        self.tools_schema = [t.get_openai_compatible_schema() for t in self.tools]

        self.max_iterations = 40
        
        
    def expand_usage(self, response: openai.ChatCompletion):
        token_details = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }

        if response.usage.completion_tokens_details:
            token_details["reasoning_tokens"] = (
                response.usage.completion_tokens_details.reasoning_tokens
            )

        if response.usage.prompt_tokens_details:
            token_details["cached_tokens"] = (
                response.usage.prompt_tokens_details.cached_tokens
            )
        return token_details

    def generate_response(self, conversation_history: List[Dict[str, Any]], **kwargs):
        
        total_token_usage = defaultdict(int)
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *conversation_history,
            ]
        else:
            messages = conversation_history

        non_changing_messages = messages.copy()
        conversation_generated = []

        for i in range(self.max_iterations):

            request_kwargs = {
                "model": self.model_name,
                "messages": non_changing_messages,
                "tools": self.tools_schema,
            }
            
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            if self.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.reasoning_effort

            response = self.client.chat.completions.create(**request_kwargs)

            token_details = self.expand_usage(response)
            for key, value in token_details.items():
                total_token_usage[key] += value

            if response.choices[0].message.tool_calls:
                non_changing_messages.append(response.choices[0].message.to_dict())
                conversation_generated.append(response.choices[0].message.to_dict())

                for tool_call in response.choices[0].message.tool_calls:

                    tool_name = tool_call.function.name
                    tool_to_call = next(
                        (t for t in self.tools if t.name == tool_name), None
                    )
                    if tool_to_call:
                        tool_args = json.loads(tool_call.function.arguments)
                        result = tool_to_call.execute(**tool_args)

                        tool_response = {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                        }

                        non_changing_messages.append(tool_response)

                        conversation_generated.append(tool_response)
                    else:
                        raise ValueError(f"Tool {tool_name} not found")
            else:
                return response.choices[0].message.content, conversation_generated, total_token_usage

        raise ValueError("Max iterations reached")


class OpenaiNativeWebSearch(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(model_name, "assistant", api_base, api_key)
        self.system_prompt = system_prompt

        self.tools=[{ "type": "web_search_preview" }]


    def generate_response(self, conversation_history: List[Dict[str, Any]], **kwargs):
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *conversation_history,
            ]
        else:
            messages = conversation_history

        non_changing_messages = messages.copy()
        conversation_generated = []


        response = self.client.responses.create(
            model=self.model_name,
            input=non_changing_messages,
            tools=self.tools,
        )

        output_messages=response.output
        
        last_message=output_messages[-1]
        
        last_message_content=last_message.content[0].text
        
        return last_message_content, []


class SubagentModel(BaseModel):
    """
    A subagent model that can answer queries using tools.
    This is used by the orchestrator to delegate smaller queries.
    """
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        tools: List[BaseTool] = [],
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        super().__init__(model_name, "assistant", api_base, api_key)
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_schema = [t.get_openai_compatible_schema() for t in self.tools]
        self.max_iterations = 40
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def generate_response(self, conversation_history: List[Dict[str, Any]], **kwargs):
        """
        Generate a response for a subagent query.
        Returns: (response_text, conversation_history)
        """
        total_token_usage = defaultdict(int)
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *conversation_history,
            ]
        else:
            messages = conversation_history

        non_changing_messages = messages.copy()
        conversation_generated = []

        for i in range(self.max_iterations):
            
            args={
                "model": self.model_name,
                "messages": non_changing_messages,
                "tools": self.tools_schema,
            }
            
            if self.temperature is not None:
                args["temperature"] = self.temperature
                
            if self.reasoning_effort:
                args["reasoning_effort"] = self.reasoning_effort
                
            response = self.client.chat.completions.create(
                **args
            )

            self.update_token_usage(response)
            token_details = self.expand_usage(response)
            for key, value in token_details.items():
                total_token_usage[key] += value

            if response.choices[0].message.tool_calls:
                non_changing_messages.append(response.choices[0].message.to_dict())
                conversation_generated.append(response.choices[0].message.to_dict())

                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_to_call = next(
                        (t for t in self.tools if t.name == tool_name), None
                    )
                    if tool_to_call:
                        tool_args = json.loads(tool_call.function.arguments)
                        result = tool_to_call.execute(**tool_args)

                        tool_response = {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                        }

                        non_changing_messages.append(tool_response)
                        conversation_generated.append(tool_response)
                    else:
                        raise ValueError(f"Tool {tool_name} not found")
            else:
                return (
                    response.choices[0].message.content,
                    conversation_generated,
                    total_token_usage,
                )

        raise ValueError("Max iterations reached")


class OrchestratorModel(BaseModel):
    """
    An orchestrator model that breaks down queries and delegates to subagents.
    Uses function calls to delegate queries to subagents.
    """
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        subagent_tool: Optional[BaseTool] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        super().__init__(model_name, "assistant", api_base, api_key)
        self.system_prompt = system_prompt
        self.subagent_tool = subagent_tool
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        if subagent_tool:
            self.tools_schema = [subagent_tool.get_openai_compatible_schema()]
        else:
            self.tools_schema = []
        
        self.max_iterations = 40

    def generate_response(self, conversation_history: List[Dict[str, Any]], **kwargs):
        """
        Generate a response by orchestrating subagents.
        Returns: (response_text, full_conversation_history)
        """
        total_token_usage = defaultdict(int)
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *conversation_history,
            ]
        else:
            messages = conversation_history

        non_changing_messages = messages.copy()
        conversation_generated = []
        full_history = []  # Reset history for this call
        full_history.append({"type": "initial_messages", "messages": messages.copy()})

        for i in range(self.max_iterations):
            args={
                "model": self.model_name,
                "messages": non_changing_messages,
                "max_completion_tokens": 6000,
                "tools": self.tools_schema,
            }
            if self.temperature is not None:
                args["temperature"] = self.temperature
            
            if self.reasoning_effort:
                args["reasoning_effort"] = self.reasoning_effort
            
            response = self.client.chat.completions.create(
                **args
            )

            self.update_token_usage(response)
            token_details = self.expand_usage(response)
            for key, value in token_details.items():
                total_token_usage[key] += value
            
            response_message = response.choices[0].message.to_dict()
            full_history.append({
                "type": "orchestrator_response",
                "iteration": i + 1,
                "message": response_message.copy()
            })

            if response.choices[0].message.tool_calls:
                non_changing_messages.append(response_message)
                conversation_generated.append(response_message)

                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    
                    if tool_name == "delegate_to_subagent" and self.subagent_tool:
                        tool_args = json.loads(tool_call.function.arguments)
                        query = tool_args.get("query", "")
                        
                        # # Track subagent call
                        # subagent_call_info = {
                        #     "type": "subagent_call",
                        #     "iteration": i + 1,
                        #     "tool_call_id": tool_call.id,
                        #     "query": query,
                        # }
                        # full_history.append(subagent_call_info)
                        
                        # Execute subagent query
                        result = self.subagent_tool.execute(**tool_args)
                        subagent_token_usage = getattr(
                            self.subagent_tool, "last_subagent_token_usage", None
                        )
                        if subagent_token_usage:
                            for key, value in subagent_token_usage.items():
                                total_token_usage[key] += value
                        
                        # Get subagent's full conversation history for debugging
                        subagent_full_history = getattr(self.subagent_tool, 'last_subagent_history', None)
                        
                        # Track subagent response with full history
                        subagent_response_info = {
                            "type": "subagent_response",
                            "tool_call_id": tool_call.id,
                            "response": result,
                            "subagent_full_history": subagent_full_history,
                            "subagent_token_usage": subagent_token_usage,
                        }
                        full_history.append(subagent_response_info)

                        tool_response = {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                        }

                        non_changing_messages.append(tool_response)
                        conversation_generated.append(tool_response)
                    else:
                        raise ValueError(f"Tool {tool_name} not found or not supported")
            else:
                final_response = response.choices[0].message.content
                full_history.append({
                    "type": "final_response",
                    "response": final_response
                })
                return final_response, full_history, total_token_usage

        raise ValueError("Max iterations reached")
