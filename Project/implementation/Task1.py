

#%% sourcing scripts

from instance_demo import * # Sourcing provided code, assuming right working directory
from Task1_Functions import * 

#%% control knobs
# Specify the number of orders we receive (either 10 or 20)
orderAmount = 10

# Specify the number of items in the warehouse (either 24 or 360)
itemAmount = 24

# From all orders, assign them to the optimal packing station:
# I use option "lite" here to get two list of orders for two packing station, as
# input for the next function. To see information of the orders, use option "full"
station = F_assignOrderToStation(10, "lite")

# For each station, create feasible batches from the orders, which also includes
# the sequence of pods visited for the cobot, as well as the minimum distance
# travelled for that batch.
batchFromStation = []

for i in range(len(station)):
    stationCopy = copy.deepcopy(station)
    packingStation = list(stationCopy.keys())[i]
    listOfOrders = list(stationCopy.values())[i]
    batchFromStation.append({"station":packingStation, "batchInfo":F_orderToBatch(listOfOrders, packingStation)})
del stationCopy
