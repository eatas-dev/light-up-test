param name string
param location string = resourceGroup().location
param applicationInsightsName string = ''
param hostingPlanId string
param tags object = {}
param mongoConnectionString string
param azureCosmosPassword string
param azureCosmosUsername string
param azureOpenAiApiKey string
param azureOpenAiEndpoint string
param azureOpenAiDeploymentName string

var functionName = 'func-${name}'
var storageAccountName = 'stg${replace(name, '-', '')}${substring(uniqueString(resourceGroup().id), 0, 5)}'

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-05-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
  }
  tags: tags
}

// Application Insightsが設定されている場合は、Application Insightsの接続文字列を設定する
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' existing = if (!empty(applicationInsightsName)) {
  name: applicationInsightsName
}

var optionalAppSettings = !empty(applicationInsightsName)
  ? [
      {
        name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
        value: 'InstrumentationKey=${applicationInsights.properties.InstrumentationKey}'
        slotSetting: false
      }
    ]
  : []

var baseAppSettings = [
  {
    name: 'AzureWebJobsStorage'
    value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
    slotSetting: false
  }
  {
    name: 'FUNCTIONS_EXTENSION_VERSION'
    value: '~4'
    slotSetting: false
  }
  {
    name: 'FUNCTIONS_WORKER_RUNTIME'
    value: 'python'
    slotSetting: false
  }
  {
    name: 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'
    value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
    slotSetting: false
  }
  {
    name: 'WEBSITE_CONTENTSHARE'
    value: toLower(functionName)
    slotSetting: false
  }
  {
    name:'AZURE_COSMOS_COLLECTION_NAME'
    value:'CosmicFoodCollection'
    slotSetting: false
  }
  {
    name:'AZURE_COSMOS_CONNECTION_STRING'
    value:mongoConnectionString
    slotSetting: false
  }
  {
    name:'AZURE_COSMOS_DB_NAME'
    value:'CosmicDB'
    slotSetting: false
  }
  {
    name:'AZURE_COSMOS_INDEX_NAME'
    value:'CosmicIndex'
    slotSetting: false
  }
  {
    name:'AZURE_COSMOS_PASSWORD'
    value:azureCosmosPassword
    slotSetting: false
  }
  {
    name:'AZURE_COSMOS_USERNAME'
    value:azureCosmosUsername
    slotSetting: false
  }
  {
    name:'AZURE_OPENAI_API_KEY'
    value:azureOpenAiApiKey
    slotSetting: false
  }
  {
    name:'AZURE_OPENAI_DEPLOYMENT_NAME'
    value:azureOpenAiDeploymentName
    slotSetting: false
  }
  {
    name:'AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME'
    value:'text-embedding'
    slotSetting: false
  }
  {
    name:'AZURE_OPENAI_EMBEDDINGS_MODEL_NAME'
    value:'text-embedding-ada-002'
    slotSetting: false
  }
  {
    name:'AZURE_OPENAI_ENDPOINT'
    value:azureOpenAiEndpoint
    slotSetting: false
  }
  {
    name:'OPENAI_API_TYPE'
    value:'azure'
    slotSetting: false
  }
  {
    name:'OPENAI_API_VERSION'
    value:'2023-09-15-preview'
    slotSetting: false
  }
]

var appSettings = union(baseAppSettings, optionalAppSettings)

resource function 'Microsoft.Web/sites@2022-03-01' = {
  name: functionName
  location: location
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: hostingPlanId
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      pythonVersion: '3.11'
      appSettings: appSettings
    }
  }

  resource config 'config' = {
    name: 'web'
    properties: {
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
    }
  }
  tags: tags
}

output functionId string = function.id
output functionName string = function.name
output functionDefaultHostName string = function.properties.defaultHostName
output identityPrincipalId string = function.identity.principalId
