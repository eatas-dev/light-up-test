targetScope = 'managementGroup'
@description('デプロイ時に業務システム管理グループに紐づけるサブスクリプションのIDを指定')
param subscriptionLandingZonesGroupId string

@description('デプロイ時に基盤管理グループに紐づけるサブスクリプションのIDを指定')
param subscriptionPlatformGroupId string

@description('デプロイ時にサンドボックス管理グループに紐づけるサブスクリプションのIDを指定')
param subscriptionSandboxGroupId string

resource landingZonesGroup 'Microsoft.Management/managementGroups@2021-04-01' = {
  scope: tenant()
  name: 'landingZones'
  properties: {
    displayName: '業務システム'
  }
}

resource platformGroup 'Microsoft.Management/managementGroups@2021-04-01' = {
  scope: tenant()
  name: 'platform'
  properties: {
    displayName: '基盤'
  }
}

resource sandboxGroup 'Microsoft.Management/managementGroups@2021-04-01' = {
  scope: tenant()
  name: 'sandbox'
  properties: {
    displayName: 'サンドボックス'
  }
}
resource landingZonesGroupSubscription 'Microsoft.Management/managementGroups/subscriptions@2020-05-01' = {
  parent: landingZonesGroup
  name: subscriptionLandingZonesGroupId
}

resource platformGroupSubscription 'Microsoft.Management/managementGroups/subscriptions@2020-05-01' = {
  parent: platformGroup
  name: subscriptionPlatformGroupId
}

resource sandboxGroupSubscription 'Microsoft.Management/managementGroups/subscriptions@2020-05-01' = {
  parent: sandboxGroup
  name: subscriptionSandboxGroupId
}
