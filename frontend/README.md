# React Frontend

## .envのセットアップ、定型文の変更

### .envファイルの準備

`/frontend/.env.example` をコピーする形で `.env` ファイルを作成してください。

| 変数名 | 用途 |
| --- | --- |
| VITE_APP_TITLE | トップ画面中央に表示される見出し |
| VITE_APP_HEADING | ナビゲーションバーに表示される見出し |
| VITE_CHAT_EXAMPLE_1 | 3つ表示される質問例の1つ目 |
| VITE_CHAT_EXAMPLE_2 | 3つ表示される質問例の2つ目 |
| VITE_CHAT_EXAMPLE_3 | 3つ表示される質問例の3つ目 |

### ロゴイメージの変更

デフォルトでは `/frontend/src/assets/FeaturedDefault.png` にロゴ画像が保存されており、ソースコード中の下記2箇所でアプリ画面内のロゴ画像が使用されています。  
必要に応じて画像ファイルの置き換えやファイル名変更を行ってください。

- /frontend/src/pages/chat/chat.tsx
- /frontend/src/components/Answer/AnswerIcon.tsx

## アプリデバッグ起動手順

This is a React Web App with a simple chat interface empowered with some developer options to perform RAG.

To install the application requirements run this command inside the `frontend/` directory:

```bash
nvm use 18
npm install
```

> [!NOTE]
> To install the development dependencies you need to use the following command:
>
> ```bash
> nvm use 18
> npm install --include=dev
> ```
>

To start the application run inside the `frontend/` directory:

```bash
npm run dev
```
