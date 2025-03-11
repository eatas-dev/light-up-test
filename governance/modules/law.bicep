targetScope = 'resourceGroup'

param name string = 'law-common'
param retentionInDays int = 30
param location string = resourceGroup().location
param actionGroupName string = 'On-call-Team'
param actionGroupEmail string

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2020-08-01' = {
  name: name
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: retentionInDays
  }
}

resource supportTeamActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: actionGroupName
  location: 'global'
  properties: {
    enabled: true
    groupShortName: actionGroupName
    emailReceivers: [
      {
        name: actionGroupName
        emailAddress: actionGroupEmail
        useCommonAlertSchema: true
      }
    ]
  }
}

output lawId string = logAnalyticsWorkspace.id
output actionGroupId string = supportTeamActionGroup.id
