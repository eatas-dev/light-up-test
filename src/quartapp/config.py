import json
import os
from collections.abc import AsyncGenerator
from uuid import uuid4

import openai

from quartapp.approaches.schemas import (
    AIChatRoles,
    Context,
    DataPoint,
    Message,
    RetrievalResponse,
    RetrievalResponseDelta,
    Thought,
)
from quartapp.config_base import AppConfigBase


class AppConfig(AppConfigBase):
    async def run_keyword(
        self, session_state: str | None, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> RetrievalResponse:
        keyword_response, answer = await self.setup.keyword.run(messages, temperature, limit, score_threshold)

        new_session_state: str = session_state if session_state else str(uuid4())

        if keyword_response is None or len(keyword_response) == 0:
            return RetrievalResponse(
                sessionState=new_session_state,
                context=Context([DataPoint()], [Thought()]),
                message=Message(content="No results found", role=AIChatRoles.ASSISTANT),
            )
        top_result = json.loads(answer)

        message_content = f"""
            Name: {top_result.get('name')}
            Description: {top_result.get('description')}
            Price: {top_result.get('price')}
            Category: {top_result.get('category')}
            Collection: {self.setup._database_setup._collection_name}
        """

        context: Context = await self.get_context(keyword_response)
        context.thoughts.insert(0, Thought(description=answer, title="Cosmos Text Search Top Result"))
        context.thoughts.insert(0, Thought(description=str(keyword_response), title="Cosmos Text Search Result"))
        context.thoughts.insert(0, Thought(description=messages[-1]["content"], title="Cosmos Text Search Query"))
        message: Message = Message(content=message_content, role=AIChatRoles.ASSISTANT)

        await self.add_to_cosmos(
            old_messages=messages,
            new_message=message.to_dict(),
            session_state=session_state,
            new_session_state=new_session_state,
        )

        return RetrievalResponse(context, message, new_session_state)

    async def run_vector(
        self, session_state: str | None, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> RetrievalResponse:
        vector_response, answer = await self.setup.vector_search.run(messages, temperature, limit, score_threshold)

        new_session_state: str = session_state if session_state else str(uuid4())

        if vector_response is None or len(vector_response) == 0:
            return RetrievalResponse(
                sessionState=new_session_state,
                context=Context([DataPoint()], [Thought()]),
                message=Message(content="No results found", role=AIChatRoles.ASSISTANT),
            )
        top_result = json.loads(answer)

        message_content = f"""
            Name: {top_result.get('name')}
            Description: {top_result.get('description')}
            Price: {top_result.get('price')}
            Category: {top_result.get('category')}
            Collection: {self.setup._database_setup._collection_name}
        """

        context: Context = await self.get_context(vector_response)
        context.thoughts.insert(0, Thought(description=answer, title="Cosmos Vector Search Top Result"))
        context.thoughts.insert(0, Thought(description=str(vector_response), title="Cosmos Vector Search Result"))
        context.thoughts.insert(0, Thought(description=messages[-1]["content"], title="Cosmos Vector Search Query"))
        message: Message = Message(content=message_content, role=AIChatRoles.ASSISTANT)

        await self.add_to_cosmos(
            old_messages=messages,
            new_message=message.to_dict(),
            session_state=session_state,
            new_session_state=new_session_state,
        )

        return RetrievalResponse(context, message, new_session_state)

    async def run_rag(
        self, session_state: str | None, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> RetrievalResponse:
        rag_response, answer = await self.setup.rag.run(messages, temperature, limit, score_threshold)
        json_answer = json.loads(answer)

        new_session_state: str = session_state if session_state else str(uuid4())

        if rag_response is None or len(rag_response) == 0:
            if answer:
                return RetrievalResponse(
                    sessionState=new_session_state,
                    context=Context([DataPoint()], [Thought()]),
                    message=Message(content=json_answer.get("response"), role=AIChatRoles.ASSISTANT),
                )
            else:
                return RetrievalResponse(
                    sessionState=new_session_state,
                    context=Context([DataPoint()], [Thought()]),
                    message=Message(content="No results found", role=AIChatRoles.ASSISTANT),
                )

        context: Context = await self.get_context(rag_response)
        context.thoughts.insert(
            0, Thought(description=json_answer.get("response"), title="Cosmos RAG OpenAI Rephrased Response")
        )
        context.thoughts.insert(
            0, Thought(description=str(rag_response), title="Cosmos RAG Search Vector Search Result")
        )
        context.thoughts.insert(
            0, Thought(description=json_answer.get("rephrased_response"), title="Cosmos RAG OpenAI Rephrased Query")
        )
        context.thoughts.insert(0, Thought(description=messages[-1]["content"], title="Cosmos RAG Query"))
        message: Message = Message(content=json_answer.get("response"), role=AIChatRoles.ASSISTANT)

        await self.add_to_cosmos(
            old_messages=messages,
            new_message=message.to_dict(),
            session_state=session_state,
            new_session_state=new_session_state,
        )

        return RetrievalResponse(context, message, new_session_state)

    async def run_rag_stream(
        self, session_state: str | None, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> AsyncGenerator[RetrievalResponseDelta, None]:
        rag_response, answer = await self.setup.rag.run_stream(messages, temperature, limit, score_threshold)

        new_session_state: str = session_state if session_state else str(uuid4())

        context: Context = await self.get_context(rag_response)
        context.thoughts.insert(
            0, Thought(description=str(rag_response), title="Cosmos RAG Search Vector Search Result")
        )
        context.thoughts.insert(0, Thought(description=messages[-1]["content"], title="Cosmos RAG Query"))

        yield RetrievalResponseDelta(context=context, sessionState=new_session_state)

        async for message_chunk in answer:
            message: Message = Message(content=str(message_chunk.content), role=AIChatRoles.ASSISTANT)
            yield RetrievalResponseDelta(
                delta=message,
            )

        await self.add_to_cosmos(
            old_messages=messages,
            new_message=message.to_dict(),
            session_state=session_state,
            new_session_state=new_session_state,
        )

    # 追加したメソッド
    async def run_gpt4o(
        self, session_state: str | None, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> RetrievalResponse:

        # 新しいセッション状態を生成（既存のものがあればそれを使用）
        new_session_state: str = session_state if session_state else str(uuid4())

        try:
            # 最後のメッセージを取得
            if messages and len(messages) > 0:
                last_message = messages[-1]
                content = last_message.get("content", "")

                # YouTubeのURLが含まれているか確認
                youtube_url = None
                for word in content.split():
                    if "youtube.com" in word or "youtu.be" in word:
                        youtube_url = word
                        break

                # YouTubeのURLが見つかった場合、YouTubeSummarizerを使用
                if youtube_url:
                    from quartapp.approaches.summarizer import YouTubeSummarizer

                    summarizer = YouTubeSummarizer()

                    documents, result_json = await summarizer.run(messages, temperature, limit, score_threshold)
                    result = json.loads(result_json)

                    if "error" in result:
                        return RetrievalResponse(
                            sessionState=new_session_state,
                            context=Context(
                                [DataPoint()],
                                [Thought(description=result["error"], title="YouTube Summarization Error")],
                            ),
                            message=Message(
                                content=(
                                    f"申し訳ありません。YouTubeの動画を評価中にエラーが発生しました: "
                                    f"{result['error']}"
                                ),
                                role=AIChatRoles.ASSISTANT,
                            ),
                        )

                    context = Context(
                        data_points=[
                            DataPoint(description=result["summary"], category="YouTube", collection=youtube_url)
                        ],
                        thoughts=[
                            Thought(description=messages[-1]["content"], title="User Query"),
                            Thought(
                                description=f"YouTube動画「{result['video_info']['title']}」の評価",
                                title="YouTube Summary",
                            ),
                            Thought(
                                description=(
                                    f"会話の整理結果: {result.get('video_info', {}).get('description', '説明なし')}"
                                ),
                                title="Video Info",
                            ),
                        ],
                    )

                    message = Message(content=result["summary"], role=AIChatRoles.ASSISTANT)

                    return RetrievalResponse(context, message, new_session_state)

            # YouTubeのURLが見つからない場合、通常のGPT-4o応答を返す
            client = openai.AsyncAzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION", "2023-09-15-preview"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )

            response = await client.chat.completions.create(
                model=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME"), messages=messages, temperature=temperature
            )

            answer_content = response.choices[0].message.content

            # 空のコンテキストを作成
            context = Context(
                data_points=[],
                thoughts=[
                    Thought(description=messages[-1]["content"], title="User Query"),
                    Thought(description="純粋なGPT-4o応答", title="GPT-4o Direct Response"),
                ],
            )

            message = Message(content=answer_content, role=AIChatRoles.ASSISTANT)

            return RetrievalResponse(context, message, new_session_state)

        except Exception as e:
            error_message = f"Error calling GPT-4o directly: {str(e)}"
            print(f"GPT-4o Error: {error_message}")  # デバッグ用
            return RetrievalResponse(
                sessionState=new_session_state,
                context=Context([DataPoint()], [Thought(description=error_message, title="Error")]),
                message=Message(
                    content=f"申し訳ありません。エラーが発生しました: {error_message}", role=AIChatRoles.ASSISTANT
                ),
            )

    async def run_gpt4o_stream(
        self, session_state: str | None, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> AsyncGenerator[RetrievalResponseDelta, None]:

        # 新しいセッション状態を生成
        new_session_state: str = session_state if session_state else str(uuid4())

        try:

            # OpenAIクライアントを新しく作成
            client = openai.AsyncAzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION", "2023-09-15-preview"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )

            # 空のコンテキストを最初に送信
            context = Context(
                data_points=[],
                thoughts=[
                    Thought(description=messages[-1]["content"], title="User Query"),
                    Thought(description="純粋なGPT-4oストリーミング応答", title="GPT-4o Direct Stream"),
                ],
            )

            yield RetrievalResponseDelta(context=context, sessionState=new_session_state)

            # ストリーミングレスポンスを作成
            stream = await client.chat.completions.create(
                model=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME"),
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            message_content = ""

            # ストリーミングレスポンスを処理
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    message_content += content
                    message = Message(content=content, role=AIChatRoles.ASSISTANT)
                    yield RetrievalResponseDelta(delta=message)

        except Exception as e:
            # エラーハンドリング
            error_message = f"Error in GPT-4o direct stream: {str(e)}"
            print(f"GPT-4o Streaming Error: {error_message}")
            error_delta = Message(
                content=f"申し訳ありません。エラーが発生しました: {error_message}", role=AIChatRoles.ASSISTANT
            )
            yield RetrievalResponseDelta(delta=error_delta)
