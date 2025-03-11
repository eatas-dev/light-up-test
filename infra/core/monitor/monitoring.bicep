metadata description = 'Creates an Application Insights instance and a Log Analytics workspace.'
param logAnalyticsSubscriptionId string
param logAnalyticsResourceGroup string
param logAnalyticsWorkspaceName string
param applicationInsightsName string
param applicationInsightsDashboardName string = ''
param location string = resourceGroup().location
param actionGroupShortName string = 'appTeam'

// Reference Properties
param emailReceivers array = [
  {
    emailAddress: 'test.user@testcompany.com'
    name: 'TestUser-EmailAction'
    useCommonAlertSchema: 'true'
  }
]

param tags object = {}
@allowed(['Enabled', 'Disabled'])
param publicNetworkAccess string = 'Enabled'

var workspaceResourceId = resourceId(
  logAnalyticsSubscriptionId,
  logAnalyticsResourceGroup,
  'Microsoft.OperationalInsights/workspaces',
  logAnalyticsWorkspaceName
)

module applicationInsights 'br/public:avm/res/insights/component:0.3.1' = {
  name: 'applicationinsights'
  params: {
    name: applicationInsightsName
    location: location
    tags: tags
    workspaceResourceId: workspaceResourceId
    publicNetworkAccessForIngestion: publicNetworkAccess
    publicNetworkAccessForQuery: publicNetworkAccess
  }
}

module applicationInsightsDashboard 'applicationinsights-dashboard.bicep' = if (!empty(applicationInsightsDashboardName)) {
  name: 'application-insights-dashboard'
  params: {
    name: applicationInsightsDashboardName
    location: location
    applicationInsightsName: applicationInsights.name
  }
}

module applicationTeamActionGroup 'br/public:avm/res/insights/action-group:0.4.0' = {
  name: 'actionGroupDeployment'
  params: {
    groupShortName: actionGroupShortName
    name: 'ApplicationTeam'
    location: 'global'
    tags: tags
    emailReceivers: emailReceivers
  }
}

output applicationInsightsConnectionString string = applicationInsights.outputs.connectionString
output applicationInsightsId string = applicationInsights.outputs.resourceId
output applicationInsightsInstrumentationKey string = applicationInsights.outputs.instrumentationKey
output applicationInsightsName string = applicationInsights.outputs.name
output logAnalyticsWorkspaceId string = workspaceResourceId
output actionGroupId string = applicationTeamActionGroup.outputs.resourceId
