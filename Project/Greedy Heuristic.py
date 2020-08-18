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

#%% CONTROL KNOBS
# Specify the number of orders we receive (either 10 or 20)
orderAmount = 10

# Specify the number of items in the warehouse (either 24 or 360)
itemAmount = 24
#%% F_itemsInOrder

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

#%% F_weightOfOrder
def F_weighOfOrder(orderID):
    totalWeight = 0
    orderInfo = F_orderInfo(orderID)
    for item in orderInfo:
        totalWeight += item["Weight"] * item["Count"]
    return(round(totalWeight), 2)

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
    distMat.insert(0, "FromPod", distMat.columns)
    distMat = distMat.set_index('FromPod')
    
    # Extract distances from pods back to the packing stations. This is the last
    # step in calculating the route.
    distToStation0 = distMat[["OutD0"]]
    distToStation1 = distMat[["OutD1"]]
    distMat = distMat.drop(["OutD0","OutD1"], axis = 1) 

#%% F_minDistance
def F_MinDist(nodesOfPods, packingStation):
    
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
    # print(float(distToStation0.loc[route[-1],:]))
    totalDist += distToStation0.loc[route[-1],:]
    totalDist = round(float(totalDist),2)
    # print("Final dist:")
    # print(totalDist)
    route.append(packingStation)
    return({"Route":route, "Distance": totalDist})
#%% F_assignOrderToStation
def F_assignOrderToStation(orderAmount, type = "full"):
    
    station = {"OutD0":[],
               "OutD1":[]}
    
    for orderID in range(orderAmount):
        itemsInOrder = F_itemsInOrder(orderID)
        # print("----Order " + str(count) +"----")
        fromStation0 = F_MinDist(itemsInOrder, "OutD0")
        # print("If start at station 0:")
        # print(fromStation0)
        fromStation1 = F_MinDist(itemsInOrder, "OutD1")
        # print("If start at station 1:")
        # print(fromStation1)
        if type == "full":
            if fromStation0["Distance"] < fromStation1["Distance"]:
                station["OutD0"].append({"orderID":orderID, "itemsInOrder" : itemsInOrder, "Distance" : fromStation0["Distance"]})
                # print("Choose station 0")
            else: 
                station["OutD1"].append({"orderID":orderID, "itemsInOrder" : itemsInOrder, "Distance" : fromStation1["Distance"]})
                # print("Choose station 1")
        elif type == "lite":
            if fromStation0["Distance"] < fromStation1["Distance"]:
                station["OutD0"].append(orderID)
            else: 
                station["OutD1"].append(orderID)
    return(station)
#%%

def F_orderToBatch(listOfOrders):
    batch = [[listOfOrders.pop(0)]]
    batchList = []
    
    while (listOfOrders != []):
        batchCopy = copy.deepcopy(batch)  
        print("batchCopy")
        print(batchCopy)
        nextOrder = listOfOrders.pop(0)
        print("Next order")
        print(nextOrder)
        update = [batch.append(nextOrder) for batch in batchCopy]
        # update = update + [nextOrder]
        print("update")
        print(update)
        batch.append(update)
        print("Batch now")
        print(batch)       
    return(batch)