import json
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIInterface:
    def __init__(self, api: Any, key: str) -> None:
        self.api_ = api
        self.system_prompt_ = f"\
            Use this JSON schema to achieve the user's goals:\n\
            {str(api)}\n\
            Respond as a list of JSON objects.\
            Do not include explanations or conversation in the response.\
        "
        self.chat_history_ = []
        self.client = OpenAI(api_key=key)

    def prompt_to_api_calls(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.8,
    ) -> List[Dict]:
        """Turns prompt into API calls.

        Args:
            prompt (str): Prompt.
            model (str, optional): OpenAI model. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): OpenAI temperature. Defaults to 0.7.

        Returns:
            Dict: API calls.
        """
        self.chat_history_.append(  # prompt taken from https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/prompts/default_prompts.py
            {
                "role": "user",
                "content": f"\
                    {prompt}\n\
                    Respond only with the output in the exact format specified in the system prompt, with no explanation or conversation.\
                ",
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": self.system_prompt_}]
                + self.chat_history_,
                temperature=temperature,
                n=1,
            )
        except Exception as e:
            print(f"Oops! Something went wrong with {e}.")
            self.chat_history_.pop()
            return []

        self.chat_history_.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

        content = self.chat_history_[-1]["content"]
        print(f"Got response:\n{content}")
        return self.post_process_response_(content)

    def post_process_response_(self, gpt_response: str) -> List[Dict]:
        """Applies some simple post-processing to the GPT response.

        Args:
            gpt_response (str): GPT response.

        Returns:
            List[Dict]: Post-processed response.
        """
        gpt_response = gpt_response.replace("'", '"')
        gpt_response = json.loads(gpt_response)

        if isinstance(gpt_response, list):
            return gpt_response
        else:
            return [gpt_response]
