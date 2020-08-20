# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 23:05:37 2020

@author: 93cha
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:54:23 2020

@author: 93cha
"""

import xml.etree.ElementTree as ET
import pandas as pd
import copy
import numpy as np
import random as rd

#%% 
def F_orderInfo(orderID):
    itemList = []
    for itemID in warehouseInstance.Orders[orderID].Positions.keys():
        weight = F_itemInfo(itemID)["Weight"]
        count = int(warehouseInstance.Orders[orderID].Positions[str(itemID)].Count)
        item = {"itemID": int(itemID), "Weight" : weight, "Count" : count}
        itemList.append(item)
    return(itemList)

#%% F_itemInfo
def F_itemInfo(itemID):
    return(warehouseInstance.ItemDescriptions[str(itemID)].__dict__)
    
#%% F_itemsInOrder
def F_itemsInOrder(orderID):
    items = [*warehouseInstance.Orders[orderID].Positions]
    items = [int(i) for i in items]
    return(items)

#%% 
def F_itemsInBatch(batchList):
    itemList = []
    for orderID in batchList:
        items = F_itemsInOrder(orderID)
        itemList = list(set(itemList) | set(items))
    return(itemList)

#%% F_weightOfOrder
def F_weightOfOrder(orderID):
    totalWeight = 0
    orderInfo = F_orderInfo(orderID)
    
    for item in orderInfo:
        totalWeight += item["Weight"] * item["Count"]
    return(round(totalWeight, 2))

#%% F_weightOfBatch
def F_weightOfOrdersInBatch(orderList):
    totalWeight = 0
    for order in orderList:
        totalWeight += F_weightOfOrder(order)
    return(totalWeight)

#%%
def F_createDistMat():
    global distToStation0
    global distToStation1
    global distMat
    
    stopAmount = int(np.sqrt(len(distance_ij)))
    distMat = np.array(list(distance_ij.values())).reshape(stopAmount, stopAmount)
    
    # Convert to pandas DataFrame
    distMat = pd.DataFrame(distMat)
    
    # Add column names
    distMat.columns = ['OutD0', 'OutD1'] + [i for i in range(0,24)]
    
    # Add an indexing column FromPod
    distMat.insert(0, "fromPod", distMat.columns)
    distMat = distMat.set_index('fromPod')
    
    # Extract distances from pods back to the packing stations. This is the last
    # step in calculating the route.
    distToStation0 = distMat[["OutD0"]]
    distToStation1 = distMat[["OutD1"]]
    distMat = distMat.drop(["OutD0","OutD1"], axis = 1) 

#%% F_minDistance
def F_minDist(nodesOfPods, packingStation):
    
    # Call for distance matrix
    F_createDistMat()    
    
    # Create destructive copy of nodesOfPods, so that each
    # node can be removed later
    currentBatch = nodesOfPods.copy()
    
    # First node of route is the chosen packing station
    route = [packingStation] 
    
    # Initialize distance
    totalDist = 0

    for i in range(len(nodesOfPods)):
        
        dist = distMat.loc[route[i],currentBatch].to_frame()
        # print("Compare distances:")
        # print(dist)
        nextNode = int(dist.idxmin())
        # print("Next optimal node:")
        # print(nextNode)
        distNode = float(dist.min())
        # print("Weight added")
        # print(distNode)
        route.append(nextNode)   
        # print("Route so far:")
        # print(route)
        totalDist += float(distNode)
        # print("Distance so far:")
        # print(totalDist)
        currentBatch.remove(nextNode)
        # print("Pods left to visit:")
        # print(currentBatch)

    # print("Distance back to packing station:")
    # print(float(distMat.loc[packingStation,route[-1]]))
    totalDist += distMat.loc[packingStation,route[-1]]
    totalDist = round(float(totalDist),2)
    # print("Final dist:")
    # print(totalDist)
    route.append(packingStation)
    return({"route":route, "distance": totalDist})

#%% F_assignOrderToStation
def F_assignOrderToStation(orderAmount, type = "full"):
    
    station = {"OutD0":[],
               "OutD1":[]}
    
    for orderID in range(orderAmount):
        itemsInOrder = F_itemsInOrder(orderID)
        # print("----Order " + str(count) +"----")
        fromStation0 = F_minDist(itemsInOrder, "OutD0")
        # print("If start at station 0:")
        # print(fromStation0)
        fromStation1 = F_minDist(itemsInOrder, "OutD1")
        # print("If start at station 1:")
        # print(fromStation1)
        if type == "full":
            if fromStation0["distance"] < fromStation1["distance"]:
                station["OutD0"].append({"orderID":orderID, "itemsInOrder" : itemsInOrder, "distance" : fromStation0["distance"]})
                # print("Choose station 0")
            else: 
                station["OutD1"].append({"orderID":orderID, "itemsInOrder" : itemsInOrder, "distance" : fromStation1["distance"]})
                # print("Choose station 1")
        elif type == "lite":
            if fromStation0["distance"] < fromStation1["distance"]:
                station["OutD0"].append(orderID)
            else: 
                station["OutD1"].append(orderID)
    return(station)
#%%
def F_orderToBatch(listOfOrders, packingStation):
    batch = []
    batch.append([listOfOrders.pop(0)])
    
    while (listOfOrders != []):
        batchCopy = copy.deepcopy(batch)  
        # print("batchCopy")
        # print(batchCopy)
        nextOrder = [listOfOrders.pop(0)]
        # print("Next order")
        # print(nextOrder)
        
        # Add nextOrder to each of the batches in batchCopy to create new batches
        batchCopy = [b + nextOrder for b in batchCopy]
        # print("batchCopy + nextOrder")
        # print(batchCopy)
        
        # Check weight capacity constraint of newly created batches
        batchCopy = [b for b in batchCopy if F_weightOfOrdersInBatch(b) <= batch_weight]
        
        # Add nextOrder to batch as a feasible batch
        batch.append(nextOrder)
        
        # Append new feasible batches in batchCopy to batch
        batch = batch + batchCopy
        # print("Batch now")
        # print(batch)   
    
    batchInfo = []    
    batchID = 0
    for b in batch:
        item = F_itemsInBatch(b)
        dist = F_minDist(item, packingStation)["distance"]
        route = F_minDist(item, packingStation)["route"]
        del route[0]
        del route[-1]
        batchInfo.append({"batchID": batchID, "ordersInBatch": b, "routeInBatch" : route, "distance": dist})
        batchID += 1
    
    batchInfo = pd.DataFrame(batchInfo)
    return(batchInfo)

#%% F_greedyHeuristic(batchInfo):
def F_greedyHeuristic(batchFromStation, packingStation):
    
    # Initialize
    greedyBatch = []
    
    # Extract orders from the station
    orderToCover = station.get(packingStation)
    
    # Extract the feasible batch information for corresponding station from 
    # batchFromStation object
    batchInfo = [batch['batchInfo'] for batch in batchFromStation if batch['station'] == packingStation]
    batchInfo = batchInfo[0]    
    
    while orderToCover != []:
        # batchInfo["numberOfBatchCovered"] = 0
        # print("Order to cover: " + str(orderToCover))
        
        # Calculate the amount of order a feasible batch can cover
        for i in list(batchInfo.index):
            value = len(set(batchInfo.loc[i,"ordersInBatch"]) & set(orderToCover))
            batchInfo.loc[i,"numberOfBatchCovered"] = value
        
        # Preliminary criteria - batch has to cover at least 1 order
        nextBatch = batchInfo.query("numberOfBatchCovered > 0")
        # print("Preliminary criteria - batch has to cover at least 1 order:")
        # print(nextBatch)
        
        # Greedy criteria 1 - subset of batches with minimum distance travelled
        nextBatch = nextBatch.query("distance == distance.min()")
        # print("Min dist criteria (greedy):")
        # print(nextBatch)
        
        # Greedy criteria 2 - subset of batches with maximum number of orders covered
        nextBatch = nextBatch.query("numberOfBatchCovered == numberOfBatchCovered.max()")
        # print("Max cover criteria:")
        # print(nextBatch)
        
        # From that subset, randomly pick a batch and append it to the final result
        random = rd.randint(0,len(nextBatch)-1)
        nextBatch = dict(nextBatch.iloc[random,:])
        # print("Choose random:")
        # print(nextBatch)
        greedyBatch.append(nextBatch)
        
        # Delete already covered orders out of orderToCover list
        orderToCover = list(set(orderToCover) - set(nextBatch["ordersInBatch"]))
        
        # Delete the already chosen batch out of the batch list
        batchToDelete = nextBatch["batchID"]
        batchInfo = batchInfo[batchInfo["batchID"] != batchToDelete]
    
    greedyBatch = pd.DataFrame(greedyBatch)
    return(greedyBatch)

#%% MAIN SCRIPT
# Source Xie's code (assuming the right working directory)
from instance_demo import *
    
# Set control knobs
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

# Final result for each of the station
greedyStation0 = F_greedyHeuristic(batchFromStation, packingStation = "OutD0")
greedyStation1 = F_greedyHeuristic(batchFromStation, packingStation = "OutD1")
