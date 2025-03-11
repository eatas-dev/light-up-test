metadata description = 'Creates an Azure Cognitive Services instance.'
param name string
param location string = resourceGroup().location
param tags object = {}
@description('The custom subdomain name used to access the API. Defaults to the value of the name parameter.')
param customSubDomainName string = name
param deployments array = []
param kind string = 'OpenAI'

@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Enabled'
param sku object = {
  name: 'S0'
}

param allowedIpRules array = []
param networkAcls object = empty(allowedIpRules)
  ? {
      defaultAction: 'Allow'
    }
  : {
      ipRules: allowedIpRules
      defaultAction: 'Deny'
    }

param logAnalyticsWorkspaceId string = ''
param vnetId string = ''
param subnetId string = ''

resource account 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: name
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  tags: tags
  kind: kind
  properties: {
    customSubDomainName: customSubDomainName
    publicNetworkAccess: publicNetworkAccess
    networkAcls: networkAcls
  }
  sku: sku
}

@batchSize(1)
resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = [
  for deployment in deployments: {
    parent: account
    name: deployment.name
    properties: {
      model: deployment.model
      raiPolicyName: contains(deployment, 'raiPolicyName') ? deployment.raiPolicyName : null
    }
    sku: contains(deployment, 'sku')
      ? deployment.sku
      : {
          name: 'Standard'
          capacity: 20
        }
  }
]

resource diagnosticSettings 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(logAnalyticsWorkspaceId)) {
  name: 'diag-${name}'
  scope: account
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [
      {
        category: 'Audit'
        enabled: true
      }
      {
        category: 'RequestResponse'
        enabled: true
      }
      {
        category: 'Trace'
        enabled: true
      }
      {
        category: 'AzureOpenAIRequestUsage'
        enabled: true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
      }
    ]
  }
}

//Create the private endpoint
resource privateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = if (!empty(vnetId) && !empty(subnetId)) {
  name: 'privatelink.openai.azure.com'
  location: 'global'
  tags: {}
  properties: {}

  resource virtualNetworkLink 'virtualNetworkLinks' = {
    name: '${privateDnsZone.name}-link'
    location: 'global'
    properties: {
      virtualNetwork: {
        id: vnetId
      }
      registrationEnabled: false
    }
  }
}

resource privateEndpointAoai 'Microsoft.Network/privateEndpoints@2023-04-01' = if (!empty(vnetId) && !empty(subnetId)) {
  name: 'pep-${name}'
  location: location
  properties: {
    subnet: {
      id: subnetId
    }
    customNetworkInterfaceName: 'pep-nic-openai'
    privateLinkServiceConnections: [
      {
        name: 'link-${name}'
        properties: {
          privateLinkServiceId: account.id
          groupIds: [
            'account'
          ]
        }
      }
    ]
  }
  tags: tags
  dependsOn: [privateDnsZone]

  resource privateDnsZoneGroup 'privateDnsZoneGroups' = {
    name: '${privateDnsZone.name}-group'
    properties: {
      privateDnsZoneConfigs: [
        {
          name: privateDnsZone.name
          properties: {
            privateDnsZoneId: privateDnsZone.id
          }
        }
      ]
    }
  }
}

output endpoint string = account.properties.endpoint
output id string = account.id
output name string = account.name
