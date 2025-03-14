# 03/12 変更点 メモ

## 1. 録画データのトランスクリプトの取得に関して

- トランスクリプトの取得には、LangChain の [Youtube Loader](https://python.langchain.com/docs/integrations/document_loaders/youtube_transcript/)を使用
- ローカルでは正常に動作させることができたが、Azure WebAPP 上での動作で失敗した
- 原因は、Youtube 側の対応により、AWS や Azure の IP アドレスからのアクセスを拒否しているため
- ※参考: 依存ライブラリ (youtube-transcript-api) の[README に記載](https://github.com/jdepoix/youtube-transcript-api)

## 2. 対応策

- Youtube Loader の代わりに、[youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)を使用してトランスクリプトを取得する方法
- [README](https://github.com/jdepoix/youtube-transcript-api#working-around-ip-bans-requestblocked-or-ipblocked-exception)に
  書かれている通り、[Webshare](https://www.webshare.io/?referral_code=w0xno53eb50g)を Proxy として設定する方法。
- 今回、時間内に実装まで間に合わなかったため、Youtube のトランスクリプト取得機能は一旦保留として、トランスクリプトをチャットフォームに入力する形で進めることにした。

## 紹介した参考ドキュメント類

- https://python.langchain.com/docs/integrations/document_loaders/youtube_transcript/
- https://python.langchain.com/docs/how_to/structured_output/
- https://github.com/jdepoix/youtube-transcript-api
