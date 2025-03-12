import json
import os
from typing import Any

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pymongo.collection import Collection

from quartapp.approaches.base import ApproachesBase


class YouTubeSummarizer(ApproachesBase):

    def __init__(
        self,
        vector_store: AzureCosmosDBVectorSearch | None = None,
        embedding: AzureOpenAIEmbeddings | None = None,
        chat: AzureChatOpenAI | None = None,
        data_collection: Collection | None = None,
        azure_openai_key: str | None = None,
        azure_openai_endpoint: str | None = None,
        azure_openai_deployment_name: str | None = None,
    ):
        if vector_store and embedding and chat and data_collection:
            super().__init__(vector_store, embedding, chat, data_collection)

        self.azure_openai_key = azure_openai_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = azure_openai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_deployment_name = azure_openai_deployment_name or os.environ.get(
            "AZURE_OPENAI_GPT4_DEPLOYMENT_NAME"
        )

        # Azure OpenAI モデルの設定
        self.llm = (
            chat
            if chat
            else AzureChatOpenAI(
                openai_api_key=self.azure_openai_key,
                azure_endpoint=self.azure_openai_endpoint,
                azure_deployment=self.azure_openai_deployment_name,
                openai_api_version=os.environ.get("OPENAI_API_VERSION", "2023-09-15-preview"),
                temperature=0.5,
            )
        )

    async def load_and_summarize_video(self, youtube_url: str) -> dict[str, Any]:
        """YouTubeの動画を読み込んで整理する"""
        try:
            # YouTubeの動画を読み込む
            loader = YoutubeLoader.from_youtube_url(
                youtube_url,
                # add_video_info=True,
                language=["ja"],
            )

            # ドキュメントを読み込む
            documents = loader.load()

            if not documents:
                print(documents)
                return {"error": f"{youtube_url}を読み込めませんでした。字幕が利用できない可能性があります。"}

            summary_prompt_template = """
            以下のYouTube動画の書き起こしをユーザーと管理栄養士に分類した会話に整理してください:
            {text}
            """

            evaluation_prompt_template = """
            以下のユーザーと管理栄養士の会話を評価してください:
            {text}
            """

            summary_prompt = PromptTemplate.from_template(summary_prompt_template)
            evaluation_prompt = PromptTemplate.from_template(evaluation_prompt_template)

            # チェーン
            summary = summary_prompt | self.llm | StrOutputParser()
            content = summary.invoke(documents[0].page_content)
            print(content)

            # 動画のメタデータ
            video_info = {
                "title": youtube_url,
                "description": content,
            }

            chain = evaluation_prompt | self.llm | StrOutputParser()
            result = chain.invoke(content)

            # 結果を返す
            return {"video_info": video_info, "summary": result, "document_chunks": len(documents)}

        except Exception as e:
            return {"error": f"要約中にエラーが発生しました: {str(e)}"}

    async def run(
        self, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> tuple[list[Document], str]:
        """ApproachesBaseのrunメソッドを実装

        最後のメッセージからYouTube URLを抽出して要約と評価を行う
        """
        # 最後のメッセージからYouTube URLを抽出
        if not messages or len(messages) == 0:
            return [], json.dumps({"error": "メッセージが空です"})

        last_message = messages[-1]
        content = last_message.get("content", "")

        # YouTube URLを抽出
        youtube_url = None
        for word in content.split():
            if "youtube.com" in word or "youtu.be" in word:
                youtube_url = word
                break

        if not youtube_url:
            return [], json.dumps({"error": "YouTubeのURLが見つかりませんでした"})

        # 動画を要約
        result = await self.load_and_summarize_video(youtube_url)

        # エラーがあった場合
        if "error" in result:
            return [], json.dumps(result)

        # Document形式に変換して返す
        documents = [
            Document(
                page_content=result["summary"],
                metadata={"title": youtube_url, "description": result["video_info"]["description"]},
            )
        ]

        return documents, json.dumps(result)

    async def summarize_video(self, youtube_url: str) -> dict[str, Any]:
        """YouTubeの動画を要約する（直接URLを指定する場合のメソッド）"""
        return await self.load_and_summarize_video(youtube_url)
