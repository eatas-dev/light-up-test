targetScope = 'managementGroup'
//スコープを管理グループに設定

//カスタムポリシーの定義名
param policyName string = 'Allow-Resources-Location'

// ポリシー定義
resource policyDefinition 'Microsoft.Authorization/policyDefinitions@2021-06-01' = {
  name: policyName
  properties: {
    displayName: '許可されている場所とリソース'
    policyType: 'Custom'
    mode: 'All'
    description: 'このポリシーでは、リソースをデプロイするときに場所とリソースの種類を制限します。'
    policyRule: {
      if: {
        not: {
          anyOf: [//japaneastでのリソース作成・globalでのリソースのいずれかに合致する場合trueを返す
            {
              allOf: [
                //リソースグループはリソースプロバイダー・リソースタイプを一致させる。in演算子で記載する
                //Japaneastへのリソースグループのリソースプロバイダー制限
                {
                  field: 'location'
                  in: ['japaneast']
                }
                {
                  field: 'type'
                  in: [
                    'Microsoft.Resources/subscriptions/resourceGroups'
                  ] 
                }
              ]
            }
            {
              allOf: [
                {
                  field: 'location'
                  in: ['japaneast']
                }
                {
                  anyOf: [
                    //Japaneastへリソースプロバイダーの制限を行う
                    //LogAnalyticsワークスペース利用のためにAzure Monitorが必要なため、リソースプロバイダーとして追加
                    //Microsoft.InsightsはApplication Insightsをjapaneastで利用する際に必要なため両方のリージョンで追加
                    //ロール付与等の際にMicrosoft.Authorization/が必要なため追加
                    { field: 'type', like: 'Microsoft.CertificateRegistration/*'}
                    { field: 'type', like: 'Microsoft.DomainRegistration/*'}
                    { field: 'type', like: 'Microsoft.Web/*'}                    
                    { field: 'type', like: 'Microsoft.CognitiveServices/*'}
                    { field: 'type', like: 'Microsoft.DocumentDB/*'}
                    { field: 'type', like: 'Microsoft.KeyVault/*'}   
                    { field: 'type', like: 'Microsoft.AlertsManagement/*'}
                    { field: 'type', like: 'Microsoft.ChangeAnalysis/*'}
                    { field: 'type', like: 'Microsoft.Intune/*'}
                    { field: 'type', like: 'Microsoft.Monitor/*'}                    
                    { field: 'type', like: 'Microsoft.OperationalInsights/*'}
                    { field: 'type', like: 'Microsoft.OperationsManagement/*'}
                    { field: 'type', like: 'Microsoft.WorkloadMonitor/*'}
                    { field: 'type', like: 'Microsoft.Insights/*'}   
                    { field: 'type', like: 'Microsoft.Authorization/*'}   
                    { field: 'type', like: 'Microsoft.Portal/*'}   
                    { field: 'type', like: 'Microsoft.Network/*'}
                    { field: 'type', like: 'Microsoft.Peering/*'}                    
                    { field: 'type', like: 'Microsoft.DataFactory/*'}           
                    { field: 'type', like: 'Microsoft.Storage/*'}
                    { field: 'type', like: 'Microsoft.StorageSync/*'}
                  ]
                }
              ]
            }
            {
              allOf: [
                {
                  field: 'location'
                  in: ['global']
                }
                {
                  anyOf: [
                    //ログアナリティクスワークスペース等グローバルにデプロイするためglobalにリージョン制限を分ける
                    { field: 'type', like: 'Microsoft.Insights/*' }
                  ]
                }
              ]
            }
          ]
        }
      }
      then: {
        effect: 'deny'
      }
    }
  }
}


// ポリシー割り当て
resource policyAssignment 'Microsoft.Authorization/policyAssignments@2021-06-01' = {
  name: policyName
  properties: {
    displayName: '許可されている場所とリソースの割り当て'
    policyDefinitionId: extensionResourceId(managementGroup().id, 'Microsoft.Authorization/policyDefinitions', policyDefinition.name)
    scope: managementGroup().id
  }
}
