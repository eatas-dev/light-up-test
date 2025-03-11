# 事前のインフラ構築手順

## 前提

- この手順を実行するユーザーはルートテナントグループのスコープで「所有者」の権限を持つこと
- Azure サブスクリプションが 3 つ以上必要です。
- デプロイには `azd` コマンドと `az` コマンドを使用します。
- 他にも node.js と python が必要で、バージョンもある程度決め打ちのため、CodeSpase もしくは DevContainer を使用します。

## 1. ガバナンス系リソースの作成

`governance` ディレクトリに移動して Bicep テンプレートをデプロイします。`cd governance` でディレクトリを移動し、以下の実行手順に従ってください。

#### 実行手順

1. 以下のコマンドでテナントを指定してログイン

   ```bash
   az login --tenant {テナントID}
   ```

2. 以下のコマンドを実行して管理グループの作成とサブスクリプションの紐づけを行う

   ```bash
   region=japaneast
   rootGroupId=<ルートテナントグループの管理グループID>
   ```

3. `main.bicepparam` にパラーメータを設定する

   ```
   param subscriptionLandingZonesGroupId = <業務システム管理グループに紐づけるサブスクリプション ID>
   param subscriptionPlatformGroupId = <基盤管理グループに紐づけるサブスクリプション ID>
   param subscriptionSandboxGroupId = <サンドボックス管理グループに紐づけるサブスクリプション ID>

   param actionGroupEmail = <全サブスクリプション共通アラートの送付先メールアドレス>
   ```

4. 以下のコマンドを実行してデプロイ

   ```bash
   az deployment mg create --template-file main.bicep \
   --location $region --management-group-id $rootGroupId \
   --parameters main.bicepparam

   ```

5. 動作確認
   以下のキャプチャの通り作成された管理グループ下にサブスクリプションが紐づいている事を確認する
   ![管理グループとサブスクリプションの紐づけ確認](../governance/docs/動作確認.png)

## 2. アプリケーション系リソースの作成

続いてアプリケーション系のインフラリソースを作成します。ルートディレクトリに戻り、以下の手順を実行してください。
なお、本番環境とステージング環境の 2 面構築する必要があるため、2 回デプロイを行います。

### 実行手順

1. `infra/main.parameters.json` の 23 行目「vnetIntegration」を確認します。`true`にすると各リソース間の Vnet 統合, PrivateLink が構成されます。
2. `infra/main.parameters.json` の 26 行目に、「運用基盤サブスクリプション ID」を記載します。
   ```
         ...
      "platformSubscriptionId": {
        "value": ""  ← ここに運用基盤サブスクリプションのIDを記載すると各種診断設定とAppInsightがONになる
                       空文字のままであればスキップされる
      },
      "lawRgName": {
        "value": "rg-common-law"
      },
      "lawName": {
        "value": "law-common"
      }
     ...
   ```
3. `infra/main.parameters.json` の 35 行目「ヘルスチェックパス」に変更があれば修正します。空文字だと AppService の正常性チェックは有効になりません。
4. `infra/main.parameters.json` の 38 行目「アラート E メール」に AppService の正常性チェックに異常があった場合に連絡する E メールアドレスを記載します。必要に応じて連絡先を追加します。

   ```
   ...
     "alertEmails": {
       "value": [
         {
           "emailAddress": "test.user@testcompany.com", ← Eメールアドレスを修正
           "name": "TestUser-EmailAction", ← ××-EmailAction
           "useCommonAlertSchema": "true" ← デフォルトtrue
         },
         // 必要に応じて以下のように送信先を追加する
         {
           "emailAddress": "example@testcompany.com",
           "name": "TestUser-EmailAction",
           "useCommonAlertSchema": "true"
         },
       ]
     }
   ...
   ```

5. 以下のコマンドでログイン

   ```bsah
   azd auth login --tenant-id {テナントID} --use-device-code
   ```

   Codespaces だとブラウザでの認証がうまく行かないので `--use-device-code` を忘れない。画面に出たコードをコピーしておき、ブラウザで入力する。

   もし上記のコマンドを入力した際に```ERROR: AADSTS50076: Due to a configuration change made by your administrator, or because you moved to a new location, you must use multi-factor authentication to access~```というようなエラーが出る場合は以下のように一度テナントIDを指定しない形式で```azd auth login```を行ってください。
   ```bsah
   azd auth login --use-device-code
   ```
   その後テナントIDを指定した形式で```azd auth login```を行います。
   ```bsah
   azd auth login --tenant-id {テナントID} --use-device-code
   ```

7. azd init でプロジェクトを初期化します。

   ```bash
   azd init
   ```

   作成する環境名を入力しますが、できるだけ他人と被らないようにしてください。

   (例: `prod<会社名><プロジェクト名>`)

8. 以下のコマンドでデプロイ

   ```bash
   azd up
   ```

   作成するサブスクリプションとリージョンを選択してデプロイをおこないます。

   - リージョン: japaneast
   - サブスクリプション: 運用基盤サブスクリプション

9. データの追加
   MongoDB に RAG 用のデータを追加します。

   - WebApp の SSH ターミナルにアクセスします。
   - 以下のコマンドを実行します。

     ```bash
     pip install -e .

     python ./scripts/add_data.py  --file="./data/food_items.json"
     ```

10. 動作確認

   webapp の URL にアクセスして動作確認を行います。

11. ステージング環境の構築
    5~9 の手順と同じ手順でステージング環境を構築します。`azd init` する際に前回の設定とは異なる環境名を入力してください。

    (例: `stg<会社名><プロジェクト名>`)

    - デプロイ先: サンドボックスサブスクリプション
