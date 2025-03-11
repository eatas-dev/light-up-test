targetScope = 'managementGroup'

param subscriptionLandingZonesGroupId string
param subscriptionPlatformGroupId string
param subscriptionSandboxGroupId string
param actionGroupEmail string

param lawCommonRgName string = 'rg-common-law'
param location string = 'japaneast'

var subscriptions = [
  {
    id: subscriptionSandboxGroupId
    name: 'sandbox'
  }
  {
    id: subscriptionLandingZonesGroupId
    name: 'landingZones'
  }
  {
    id: subscriptionPlatformGroupId
    name: 'platform'
  }
]

var activityLogAlertRgName = 'rg-alert'

/*
  Deploy the management group
  deploy scope is the root management group
*/
module managementGroup './modules/mg.bicep' = {
  name: 'managementGroupDeploy'
  params: {
    subscriptionLandingZonesGroupId: subscriptionLandingZonesGroupId
    subscriptionPlatformGroupId: subscriptionPlatformGroupId
    subscriptionSandboxGroupId: subscriptionSandboxGroupId
  }
}

/*
  Deploy the common resource group
  deploy scope is the platformSubscription
*/
module comonRG 'modules/rg.bicep' = {
  name: 'commonRgDeploy'
  scope: subscription(subscriptionPlatformGroupId)
  params: {
    name: lawCommonRgName
    location: location
  }
  dependsOn: [managementGroup]
}

/*
  Deploy the LogAnalyticsWorkspace
  deploy scope is the common resource group
*/
module law 'modules/law.bicep' = {
  name: 'commonLawDeploy'
  scope: resourceGroup(subscriptionPlatformGroupId, lawCommonRgName)
  params: {
    name: 'law-common'
    retentionInDays: 30
    location: location
    actionGroupEmail: actionGroupEmail
  }
  dependsOn: [comonRG]
}

/*
  Deploy the activity log alert resource group
  deploy scope is the 3 subscription
*/
module activityLogAlertRGs 'modules/rg.bicep' = [
  for sub in subscriptions: {
    name: '${sub.name}AlertRgDeploy'
    scope: subscription(sub.id)
    params: {
      name: activityLogAlertRgName
      location: location
    }
    dependsOn: [law]
  }
]

/*
  Deploy the activity log alert
  deploy scope is the 3 activity log alert resource group
*/
module activityLogAlerts 'modules/activityLogAlert.bicep' = [
  for sub in subscriptions: {
    name: '${sub.name}ActivityLogAlertDeploy'
    scope: resourceGroup(sub.id, activityLogAlertRgName)
    params: {
      actionGroupId: law.outputs.actionGroupId
    }
    dependsOn: [activityLogAlertRGs]
  }
]

module policy 'modules/policy.bicep' = {
  name: 'policyDeploy'
  params: {
    policyName: 'Allow-Resources-Location'
  }
  dependsOn: [managementGroup]
}

output logAnalyticsWorkspaceId string = law.outputs.lawId
