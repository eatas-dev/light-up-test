metadata description = 'Azure Cosmos DB MongoDB vCore cluster'
@maxLength(40)
param name string
param location string = resourceGroup().location
param tags object = {}

@description('Username for admin user')
param administratorLogin string
@secure()
@description('Password for admin user')
@minLength(8)
@maxLength(128)
param administratorLoginPassword string
@description('Whether to allow all IPs or not. Warning: No IP addresses will be blocked and any host on the Internet can access the coordinator in this server group. It is strongly recommended to use this rule only temporarily and only on test clusters that do not contain sensitive data.')
param allowAllIPsFirewall bool = false
@description('Whether to allow Azure internal IPs or not')
param allowAzureIPsFirewall bool = false
@description('IP addresses to allow access to the cluster from')
param allowedSingleIPs array = []
@description('Mode to create the mongo cluster')
param createMode string = 'Default'
@description('Whether high availability is enabled on the node group')
param highAvailabilityMode bool = false
@description('Number of nodes in the node group')
param nodeCount int
@description('Deployed Node type in the node group')
param nodeType string = 'Shard'
@description('SKU defines the CPU and memory that is provisioned for each node')
param sku string
@description('Disk storage size for the node group in GB')
param storage int

@description('Log Analytics Workspace ID for diagnostic settings')
param logAnalyticsWorkspaceId string = ''

param vnetId string = ''
param subnetId string = ''

resource mognoCluster 'Microsoft.DocumentDB/mongoClusters@2024-02-15-preview' = {
  name: name
  tags: tags
  location: location
  properties: {
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorLoginPassword
    createMode: createMode
    nodeGroupSpecs: [
      {
        diskSizeGB: storage
        enableHa: highAvailabilityMode
        kind: nodeType
        nodeCount: nodeCount
        sku: sku
      }
    ]
  }

  resource firewall_all 'firewallRules' = if (allowAllIPsFirewall) {
    name: 'allow-all-IPs'
    properties: {
      startIpAddress: '0.0.0.0'
      endIpAddress: '255.255.255.255'
    }
  }

  resource firewall_azure 'firewallRules' = if (allowAzureIPsFirewall) {
    name: 'allow-all-azure-internal-IPs'
    properties: {
      startIpAddress: '0.0.0.0'
      endIpAddress: '0.0.0.0'
    }
  }

  resource firewall_single 'firewallRules' = [
    for ip in allowedSingleIPs: {
      name: 'allow-single-${replace(ip, '.', '')}'
      properties: {
        startIpAddress: ip
        endIpAddress: ip
      }
    }
  ]
}

resource diagnosticSettings 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = if (!empty(logAnalyticsWorkspaceId)) {
  name: 'diag-${name}'
  scope: mognoCluster
  properties: {
    workspaceId: logAnalyticsWorkspaceId
    logs: [
      {
        category: 'vCoreMongoRequests'
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
  name: 'privatelink.mongocluster.cosmos.azure.com'
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

resource privateEndpointMongo 'Microsoft.Network/privateEndpoints@2023-04-01' = if (!empty(vnetId) && !empty(subnetId)) {
  name: 'pep-${name}'
  location: location
  properties: {
    subnet: {
      id: subnetId
    }
    customNetworkInterfaceName: 'pep-nic-mongo'
    privateLinkServiceConnections: [
      {
        name: 'link-${name}'
        properties: {
          privateLinkServiceId: mognoCluster.id
          groupIds: [
            'MongoCluster'
          ]
        }
      }
    ]
  }
  tags: tags
  dependsOn: [privateDnsZone]

  resource privateDnsZoneGroup 'privateDnsZoneGroups' = {
    name: '${privateEndpointMongo.name}-group'
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

output connectionStringKey string = mognoCluster.properties.connectionString
