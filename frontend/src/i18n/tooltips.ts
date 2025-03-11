// Keep values less than 20 words.
// Don't add links to the tooltips.
export const toolTipText = {
    // promptTemplate: "Overrides the prompt used to generate the answer based on the question and search results."
    promptTemplate: "質問と検索結果に基づいて回答を生成するためのプロンプトを上書きします。",
    //temperature: "Sets the temperature of the request to the LLM that generates the answer. Higher temperatures result in more creative responses, but they may be less grounded.",
    temperature: "リクエストの創造性を設定します。創造性が高いほど、より創造的な応答が生成されますが、現実味が薄れる可能性があります。",
    searchScore: "Azure Cosmos DB for MongoDB vCoreから返される検索結果の最小スコアを設定します。",
    retrieveNumber:
        "Azure Cosmos DB for MongoDB vCoreから取得する検索結果の数を設定します。結果が多いほど正しい回答が見つかる可能性が高まりますが、モデルが「途中で迷子になる」可能性もあります。",
    retrievalMode:
        "Azure Cosmos DB for MongoDB vCoreクエリの検索モードを設定します。「RAG with Vector Search」はベクトル検索とLLMのリフレーズを組み合わせて使用し、「Vectors」はベクトル検索のみを使用し、「Text」は全文検索のみを使用します。",
    streamChat: "生成されると同時にチャットUIに応答をストリーミングします。"
};
