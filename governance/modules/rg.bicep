targetScope = 'subscription'

param name string
param location string = 'japaneast'

resource commonResourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: name
  location: location
}

output commonResourceGroupName string = commonResourceGroup.name
