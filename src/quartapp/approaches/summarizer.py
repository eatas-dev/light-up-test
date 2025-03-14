import json
import os
from typing import Any

# from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr
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
                api_key=SecretStr(self.azure_openai_key) if self.azure_openai_key else None,
                azure_endpoint=self.azure_openai_endpoint,
                azure_deployment=self.azure_openai_deployment_name,
                model_version=os.environ.get("OPENAI_API_VERSION", "2023-09-15-preview"),
                temperature=0.7,
            )
        )

    async def load_and_summarize_video(self, youtube_url: str) -> dict[str, Any]:
        """YouTubeの動画を読み込んで整理する"""
        try:
            # youtubeから文字起こしを取得したかったが、以下の制限により、デプロイ環境ではBlokされるため、
            # 今回はテキストから推論するように。
            # https://github.com/jdepoix/youtube-transcript-api
            # loader = YoutubeLoader.from_youtube_url(
            #     youtube_url,
            #     # add_video_info=True,
            #     language=["ja"])

            # # ドキュメントを読み込む
            # documents = loader.load()
            documents = youtube_url

            evaluation_prompt_template = """
            あなたは管理栄養士と食習慣改善を目指すユーザー間の会話を分析する専門アシスタントです。会話の内容を分析し、以下の形式でアウトプットを生成してください。

            管理栄養士と食習慣を改善したいユーザーとの会話です。
            2者の会話を分けたうえで、面談のポイントをまとめてください。
            また以下の観点で管理栄養士が面談を進行できているかを評価してください。アウトプットは各項目に対してはtrue,
            falseで返してください。
            会話を全て記載する必要はなく、該当する文章のみあわせて記載してください。
            管理栄養士とユーザーの会話の比率を計算してください。

            ###項目
            - フィードバック:事実を客観的にとらえ、ユーザーに事実を伝えている
            - メタコミュニケーション:現在交わしているコミュニケーションが効果的かどうかを確認することです。
            コーチングを効果的に進めるために、コーチとクライアントが距離をもって会話を振り返り、お互いに利益を生み出しているかどうかを確認します。
            【メタコミュニケーションの場面】
            * 会話が堂々巡りになっていると感じたとき
            * 会話にズレを感じたとき
            * 具体的な行動を決めたとき
            【メタコミュニケーションの確認内容】
            * コーチの思い込みでコーチングが行われていないかどうか
            * コーチングの振り返りを行い、順調にコーチングができているかどうかの確認
            * コーチングが機能しているか
            【メタコミュニケーションの例】
            * 「ここまで話してどうですか？」
            - 目標設定:具体的な行動目標の設定ができているか、その行動目標に対してユーザーの了承を得ているか
            - ビジュアライズ:達成を見えるようにすること。①〜⑤のどの種類のビジュアライズができているかも評価してください
            ①フューチャー・ペーシング
            ②モデルになる人に置き換える（モデリング）
            ③ディソシエート / アソシエート
            ④リソース:うまくいくイメージのもとになる資源・材料のこと。
            過去にうまくいった経験が役に立つこと。過去のダイエット経験、良い行動をした際の体験
            ⑤間接法
            - 具体的な提案:何をどしたらよいか理由とともに、どのシーン、どのタイミングでどんな行動をしたらよいか。
            ユーザーが実行したいと思う提案ができているか。
            - アクノレッジメント:相手の変化や成果に気づき、それを言語化してはっきり伝えること。
            単にほめることではない。承認する、相手の存在を認める、相手に感謝する、信頼してまかせること。
            -傾聴を妨げる要因（ブロッキング）がない:勝手な推測、先入観、思い込み、興味関心、同一視、評価・批判、競争意識
            -ユーザーの返答を受容しているか
            -ユーザーの気持ちを汲み取っているか
            -ユーザーの気持ちを無視していないか

            全体を通して良い点とよりよくするための改善点をまとめてください。

            以下を参考に全体を通しての評価を点数化してください。
            ###点数化項目(コーチングの項目に漏れがないか。)
            - フィードバック:5回以上10点、3〜4回5点、1〜2回：3点、0回0点
            - メタコミュニケーション:3回以上10点、2回以上5点、1回3点、0回0点
            - 目標設定:3個以内10点、5個以上3点
            - ビジュアライズ:3回以上10点、1〜2回5点、0回0点
            - 具体的な提案（シーンとタイミングと実行可能な提案）:10点: 抽象的な場合は5点
            - アクノレッジメント:10点
            -傾聴を妨げる要因（ブロッキング）がない:10点
            -ユーザーの実行意欲が湧く提案をしている:10点
            -専門家として自信のある対応をしている:10点
            -会話比率は管理栄養士50%未満である:10点

            ###抽出指示
            各点数化項目について、会話のどの部分が評価に反映されたのか具体的に抜き出して記載してください。例えば：

            - フィードバック（X点）:
            * 「[管理栄養士の発言をそのまま引用]」
            * 「[管理栄養士の別の発言を引用]」
            * （他の該当発言も同様に列挙）

            - メタコミュニケーション（X点）:
            * 「[該当する発言を引用]」
            * （他の該当発言も同様に列挙）

            （以下、各点数化項目について同様に記載）

            それぞれの項目について、点数とその根拠となった会話部分を明確に関連付けて記載してください。引用は正確に行い、会話のコンテキストがわかるよう十分な長さを確保してください。


            ----
            {text}
            """

            evaluation_prompt = PromptTemplate.from_template(evaluation_prompt_template)

            # 動画のメタデータ
            video_info = {
                "title": youtube_url,
                "description": "youtube video",
            }

            chain = evaluation_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": documents})
            return {"video_info": video_info, "summary": result, "document_chunks": len(documents)}

        except Exception as e:
            return {"error": f"要約中にエラーが発生しました: {str(e)}"}

    async def run(
        self, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> tuple[list[Document], str]:
        """ApproachesBaseのrunメソッドを実装

        最後のメッセージからYouTube URLを抽出して要約と評価を行う
        """
        # 最後のメッセージからYouTube URLを抽出 (最後のメッセージを使うことに急遽変更)
        if not messages or len(messages) == 0:
            return [], json.dumps({"error": "メッセージが空です"})

        last_message = messages[-1]
        content = last_message.get("content", "")
        youtube_url = content

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
