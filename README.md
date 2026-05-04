# mnemos-bridge-gemini

Google Gemini adapter for the `mnemos-bridge-core` MCP bridge abstraction. It translates MCP tool definitions into Gemini `functionDeclarations` and dispatches Gemini `functionCall` parts back to MCP tools.

## Install

```bash
pip install mnemos-bridge-gemini
```

## Quick Start

```python
import google.generativeai as genai

from mnemos_bridge_gemini import MnemosGeminiAdapter


async def main() -> None:
    genai.configure(api_key="...")

    adapter = await MnemosGeminiAdapter.connect(
        "http://192.168.207.67:5003",
        "...mcp token...",
    )

    try:
        model = genai.GenerativeModel(
            "gemini-3-pro-preview",
            tools=await adapter.gemini_tools(),
        )
        chat = model.start_chat()

        response = chat.send_message("Search MNEMOS for memories about infrastructure")
        function_call = response.candidates[0].content.parts[0].function_call

        function_response = await adapter.handle_function_call(function_call)
        final = chat.send_message(
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(**function_response)
            )
        )
        print(final.text)
    finally:
        await adapter.aclose()
```

## Multi-Turn Example

```python
import google.generativeai as genai

from mnemos_bridge_gemini import MnemosGeminiAdapter


async def run_loop(prompt: str) -> str:
    async with await MnemosGeminiAdapter.connect(
        "http://192.168.207.67:5003",
        "...mcp token...",
    ) as adapter:
        model = genai.GenerativeModel(
            "gemini-3-pro-preview",
            tools=await adapter.gemini_tools(),
        )
        chat = model.start_chat()
        response = chat.send_message(prompt)

        for _ in range(8):
            parts = response.candidates[0].content.parts
            function_calls = [
                part.function_call
                for part in parts
                if getattr(part, "function_call", None)
                and getattr(part.function_call, "name", "")
            ]
            if not function_calls:
                return response.text

            response_parts = []
            for function_call in function_calls:
                function_response = await adapter.handle_function_call(function_call)
                response_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            **function_response
                        )
                    )
                )
            response = chat.send_message(response_parts)

        return response.text
```

## Gemini Schema Subset

Gemini function parameters use a strict JSON Schema subset. This adapter flattens local nested `$ref` references before stripping keywords Gemini does not accept:

`additionalProperties`, `oneOf`, `not`, `anyOf`, `allOf`, `if`, `then`, `else`, `patternProperties`, `definitions`, `$defs`, `contentEncoding`, `contentMediaType`, `deprecated`, `readOnly`, and `writeOnly`.

When `mnemos_bridge_core.SchemaTranslator.to_gemini()` is available, the adapter delegates schema translation to it and then applies the same conservative cleanup pass. See the Gemini function calling documentation: https://ai.google.dev/gemini-api/docs/function-calling

## Vertex AI Usage

The same Gemini SDK flow can be used with Vertex AI credentials. Set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON file and configure the SDK for Vertex AI:

```python
import google.generativeai as genai

genai.configure(vertexai=True)
```

Then construct `GenerativeModel` and pass `tools=await adapter.gemini_tools()` as in the examples above.

## Testing

Run offline unit tests without real Google or MNEMOS services:

```bash
pytest tests/test_translator_offline.py tests/test_handle_function_call_offline.py
```

Run the guarded integration test by setting all required service environment variables:

```bash
export GOOGLE_API_KEY=...
export MNEMOS_TEST_BASE=http://192.168.207.67:5003
export MNEMOS_MCP_TOKEN=...
export MNEMOS_TEST_GEMINI_MODEL=gemini-3-pro-preview
pytest tests/integration/test_gemini_tool_loop.py
```
