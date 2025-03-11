param vnetName string
param location string = resourceGroup().location
param tags object = {}

var webappSubnetName = 'subnet-webapp'
var pepSubnetName = 'subnet-pv'
var vnetAddressPrefix = '10.0.0.0/16'
var webAppSubnetAddressPrefix = '10.0.1.0/24'
var pepSubnetAddressPrefix = '10.0.2.0/24'

resource nsgPrivateEndpoint 'Microsoft.Network/networkSecurityGroups@2024-01-01' = {
  name: 'nsg-pep'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowBackendSubnetInbound'
        type: 'Microsoft.Network/networkSecurityGroups/securityRules'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: webAppSubnetAddressPrefix
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
          sourcePortRanges: []
          destinationPortRanges: []
          sourceAddressPrefixes: []
          destinationAddressPrefixes: []
        }
      }
      {
        name: 'DenyVnetInbound'
        type: 'Microsoft.Network/networkSecurityGroups/securityRules'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: 'VirtualNetwork'
          access: 'Deny'
          priority: 4096
          direction: 'Inbound'
          sourcePortRanges: []
          destinationPortRanges: []
          sourceAddressPrefixes: []
          destinationAddressPrefixes: []
        }
      }
    ]
  }
  tags: tags
}

resource vnet 'Microsoft.Network/virtualNetworks@2020-06-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressPrefix
      ]
    }
    subnets: [
      {
        name: webappSubnetName
        properties: {
          addressPrefix: webAppSubnetAddressPrefix
          delegations: [
            {
              name: 'delegation'
              properties: {
                serviceName: 'Microsoft.Web/serverFarms'
              }
            }
          ]
        }
      }
      {
        name: pepSubnetName
        properties: {
          addressPrefix: pepSubnetAddressPrefix
          privateEndpointNetworkPolicies: 'NetworkSecurityGroupEnabled'
          privateLinkServiceNetworkPolicies: 'Enabled'
          networkSecurityGroup: {
            id: nsgPrivateEndpoint.id
          }
        }
      }
    ]
  }
  tags: tags
}

output vnetId string = vnet.id
output webappSubnetId string = resourceId('Microsoft.Network/virtualNetworks/subnets', vnetName, webappSubnetName)
output pepSubnetId string = resourceId('Microsoft.Network/virtualNetworks/subnets', vnetName, pepSubnetName)
