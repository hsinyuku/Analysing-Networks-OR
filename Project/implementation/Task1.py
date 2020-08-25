

#%% sourcing scripts

from instance_demo import * # Sourcing provided code, assuming right working directory
from Task1_Functions import * 

#%% control knobs
# Specify the number of orders we receive (either 10 or 20)
orderAmount = 10

# Specify the number of items in the warehouse (either 24 or 360)
itemAmount = 24

#%%
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
        # calls the name of the packing station
    listOfOrders = list(stationCopy.values())[i]
        # calls the orders that belong to that station
    batchFromStation.append({"station":packingStation, "batchInfo":F_orderToBatch(listOfOrders, packingStation)})
        # assign orders from list of orders to batches, procuding all feasible
        # batches for each station
del stationCopy
#%%

# Final result for each of the station: for each station, take all feasible 
# batches and run the greedy heuristic to choose the optimal sequence of 
# batches
greedyStation0 = F_greedyHeuristic(batchFromStation, packingStation = "OutD0")
greedyStation1 = F_greedyHeuristic(batchFromStation, packingStation = "OutD1")

#%% Task 2.2
# results of delete and repair from random neibour

removedBatch0, removedOrder0, remainSol0  = F_randomDelSol(greedyStation0)
pickedBatch0, pickedOrder0,pickedBatchInfo0 = F_randomPickBatch(batchFromStation, "OutD0",removedOrder0)
F_newSol(greedyStation0, removedBatch0, removedOrder0, remainSol0, pickedBatch0, pickedOrder0, pickedBatchInfo0)

removedBatch1, removedOrder1, remainSol1  = F_randomDelSol(greedyStation1) 
pickedBatch1, pickedOrder1, pickedBatchInfo1 = F_randomPickBatch(batchFromStation, "OutD1",removedOrder1)
F_newSol(greedyStation1,removedBatch1, removedOrder1, remainSol1, pickedBatch1, pickedOrder1, pickedBatchInfo1)