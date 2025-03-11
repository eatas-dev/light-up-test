targetScope = 'resourceGroup'

param activityLogAlertName string = '${uniqueString(resourceGroup().id)}-activitylog-alert'
param serviceHealthalertName string = '${uniqueString(resourceGroup().id)}-servicehealth-alert'
param actionGroupId string

//アクティビティログアラート
//重大なアクティビティログのみアラートを出すためcontainsAnyを追加

resource activityLogAlert 'Microsoft.Insights/activityLogAlerts@2023-01-01-preview' = {
  name: activityLogAlertName
  location: 'global'
  properties: {
    enabled:true
    condition: {
      allOf: [
        {
          field: 'category'
          equals: 'Administrative'
        }
        {
          field: 'operationName'
          equals: 'Microsoft.Resources/deployments/write'
        }
        {
          field: 'resourceType'
          equals: 'Microsoft.Resources/deployments'
        }
        {
          field: 'level'
          containsAny: [
            'critical'
            'error'
            'warning'
          ]
        }
      ]
    }
    actions: {
      actionGroups: [
        {
          actionGroupId: actionGroupId
        }
      ]
    }
    scopes: [
      subscription().id
    ]
  }
}

//サービスヘルスアラートを追加

resource serviceHealthAlert 'Microsoft.Insights/activityLogAlerts@2023-01-01-preview' = {
  name: serviceHealthalertName
  location: 'global'
  properties: {
    enabled:true
    condition: {
      allOf: [
        {
          field: 'category'
          equals: 'ServiceHealth'
        }
        {
          field: 'properties.impactedServices[*].ImpactedRegions[*].RegionName'
          containsAny: [
            'japaneast'
            'Japan East'
            'global'
            'Global'
          ]
        }
      ]
    }
    actions: {
      actionGroups: [
        {
          actionGroupId: actionGroupId
        }
      ]
    }
    scopes: [
      subscription().id
    ]
  }
}
