param name string
param location string = resourceGroup().location
param tags object = {}

param principalId string = ''
param logAnalyticsWorkspaceId string = ''
param vnetId string = ''
param subnetId string = ''

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    accessPolicies: !empty(principalId)
      ? [
          {
            objectId: principalId
            permissions: { secrets: ['get', 'list'] }
            tenantId: subscription().tenantId
          }
        ]
      : []
  }
}

resource diagnosticSettings 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(logAnalyticsWorkspaceId)) {
  name: 'diag-${name}'
  scope: keyVault
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [
      {
        category: 'AuditEvent'
        enabled: true
      }
      {
        category: 'AzurePolicyEvaluationDetails'
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
  name: 'privatelink.vaultcore.azure.net'
  location: 'global'
  properties: {}

  resource privateDnsZoneLink 'virtualNetworkLinks' = {
    name: '${privateDnsZone.name}-link'
    location: 'global'
    properties: {
      registrationEnabled: false
      virtualNetwork: {
        id: vnetId
      }
    }
  }
}

resource privateEndpointKv 'Microsoft.Network/privateEndpoints@2021-05-01' = if (!empty(vnetId) && !empty(subnetId)) {
  name: 'pep-${name}'
  location: location
  properties: {
    subnet: {
      id: subnetId
    }
    customNetworkInterfaceName: 'pep-nic-kv'
    privateLinkServiceConnections: [
      {
        name: 'link-${name}'
        properties: {
          privateLinkServiceId: keyVault.id
          groupIds: [
            'vault'
          ]
        }
      }
    ]
  }
  tags: tags
  dependsOn: [privateDnsZone]

  resource privateDnsZoneGroup 'privateDnsZoneGroups' = {
    name: '${privateEndpointKv.name}-group'
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

output endpoint string = keyVault.properties.vaultUri
output name string = keyVault.name
