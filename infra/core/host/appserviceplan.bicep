metadata description = 'Creates an Azure App Service plan with diagnostic settings.'
param name string
param location string = resourceGroup().location
param tags object = {}

param kind string = ''
param reserved bool = true
param sku object

param logAnalyticsWorkspaceId string = ''
param enableMetrics bool = true

resource appServicePlan 'Microsoft.Web/serverfarms@2022-03-01' = {
  name: name
  location: location
  tags: tags
  sku: sku
  kind: kind
  properties: {
    reserved: reserved
  }
}

// App Service Planの診断設定
resource diagnosticSettings 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(logAnalyticsWorkspaceId)) {
  name: 'diag-${name}'
  scope: appServicePlan
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    metrics: enableMetrics
      ? [
          {
            category: 'AllMetrics'
            enabled: true
          }
        ]
      : []
  }
}

output id string = appServicePlan.id
output name string = appServicePlan.name
output diagnosticsName string = !empty(logAnalyticsWorkspaceId) ? diagnosticSettings.name : ''
