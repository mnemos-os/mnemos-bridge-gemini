import os

import pytest


pytestmark = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and os.getenv("MNEMOS_TEST_BASE")),
    reason="requires GOOGLE_API_KEY and MNEMOS_TEST_BASE",
)


@pytest.mark.asyncio
async def test_gemini_tool_loop_requests_search_memories():
    import google.generativeai as genai

    from mnemos_bridge_gemini import MnemosGeminiAdapter

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    adapter = await MnemosGeminiAdapter.connect(
        os.environ["MNEMOS_TEST_BASE"],
        os.getenv("MNEMOS_MCP_TOKEN", ""),
    )

    try:
        model = genai.GenerativeModel(
            os.getenv("MNEMOS_TEST_GEMINI_MODEL", "gemini-3-pro-preview"),
            tools=await adapter.gemini_tools(),
        )
        chat = model.start_chat()

        response = chat.send_message("Search MNEMOS for memories about infrastructure")
        parts = response.candidates[0].content.parts
        function_calls = [
            part.function_call
            for part in parts
            if getattr(part, "function_call", None)
            and getattr(part.function_call, "name", "")
        ]

        assert any(function_call.name == "search_memories" for function_call in function_calls)

        function_call = next(
            function_call
            for function_call in function_calls
            if function_call.name == "search_memories"
        )
        function_response = await adapter.handle_function_call(function_call)
        final = chat.send_message(
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(**function_response)
            )
        )

        final_text = final.text
        assert final_text
        assert "memory" in final_text.lower() or "infrastructure" in final_text.lower()
    finally:
        await adapter.aclose()
