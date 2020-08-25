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

#%% CREATE ITEM LIST --> object itemList
# This code block extracts the item descriptions from XML file to a dictionary,
# put them in a list, and removes all surplus information.

# Parse XML file to tree element
tree = ET.parse("data/sku" + str(itemAmount) + "/orders_" + str(orderAmount) +
                "_mean_5_sku_" + str(itemAmount) + ".xml")

# Get the root node
root = tree.getroot()

# Create empty dictionary
itemList = list()

# Loop through child node "ItemDescription" and append values to itemList
count = 0
for itemDes in root.iter("ItemDescription"):
    temp = itemDes.attrib
    itemList.append(temp)
    count = count + 1  

# From the itemList, keep only ID and Weight
for i in itemList:
    i["itemID"] = int(i.pop("ID"))
    i["Weight"] = float(i.pop("Weight"))
    i.pop("Color",None)
    i.pop("Letter",None)
    i.pop("Type",None)

# Print final itemList:
    repr(itemList)
    
#%% Extract the order descriptions from XML file to a dictionary
orderList = list()
   
for i in range(len(root[2])):
    orderList.append(list())
    for j in range(len(root[2][i][0])):
        orderList[i].append(root[2][i][0][j].attrib)

# Rename key itemID
for order in orderList:
    for item in order:
        item["itemID"] = int(item.pop("ItemDescriptionID"))
        item["Count"] = int(item.pop("Count"))
           
# Fill the weight information of each item to the orderList
for i in range(len(orderList)):
    for j in range(len(orderList[i])):
        itemID = orderList[i][j]["itemID"]
        orderList[i][j]["Weight"] = float(itemList[int(itemID)]["Weight"])

#%% Calculate total weight of each order
    orderWeight = list()
    
    for i in range(len(orderList)):
        orderWeight.append(sum(item['Weight'] for item in orderList[i]))

#%% Function: Calculate total weight of a potential batch
    def FUNC_batchWeight(batch, orderWeight):
        sum = 0
        for order in batch:
            sum = sum + orderWeight[order]
        return(sum)
    
#%% Calculate feasible batches from the orders, only accept batches which are 
# within the cobot's capacity
   
orderSet = list(range(0,orderAmount))
feasibleBatch = list()
feasibleBatch.append([orderSet.pop(0)])
feasibleBatchWithWeight = list()

for i in range(len(orderSet)):

    feasibleBatchCopy = copy.deepcopy(feasibleBatch)    

    update = [batch + [orderSet[i]] for batch in feasibleBatchCopy]

    feasibleBatchCopy = copy.deepcopy(update)
    feasibleBatchCopy.append([orderSet[i]])
    
    # Check for cobot capacity fullfilment
    feasibleBatchCheckWeight = list()
    
    for batch in feasibleBatchCopy:
        weight = round(FUNC_batchWeight(batch, orderWeight),2)
        print("Potential batch is: " + str(batch) + ", Weight = " + str(weight))
        if (weight < 18):
            print("Within capacity. Feasible batch.")
            feasibleBatchCheckWeight.append(batch)
            feasibleBatchWithWeight.append({"Batch" : batch, "Weight" : weight})
        else:
            print("Cobot capacity exceeded. Remove from list.")
        print("---")
    feasibleBatch = feasibleBatch + feasibleBatchCheckWeight
      
#%%
feasibleBatchWeight = list()
for batch in feasibleBatch:
    if (FUNC_batchWeight(batch, orderWeight) < 10):
        feasibleBatchWeight.append(batch)

#%% 
itemDict = {}
for i in itemList:
    itemDict.append(i)
        
#%% Navigate through the itemList and print Weight
for item in itemList:
    if item["itemID"] == "5":
        print(item["Weight"])
        
                
#%% From orderList, construct itemsInOrderList
itemsInOrderList = []
for i in range(orderAmount):
    itemsInOrder = []
    for order in orderList[i]:
        if (order["itemID"] in itemsInOrder) == False:
            itemsInOrder.append(order["itemID"])
    itemsInOrderList.append(itemsInOrder)
    
#%% From itemsinOrderList and feasibleBatch, contruct itemsinBatch
itemsInBatch = []
for batch in feasibleBatch:
    print("Batch" + str(batch))
    itemsInOneBatch = []
    for i in range(len(batch)):
        order = batch[i]
        items = itemsInOrderList[order]
        itemsInOneBatch = list(set(items) | set(itemsInOneBatch))
        print("itemsInBatch" + str(itemsInOneBatch))
    itemsInBatch.append(itemsInOneBatch)
    
#%% Create a batchList 
batchList = []
for i in range(len(feasibleBatch)):    
    batch = {"batchID": i, "ordersInBatch":feasibleBatch[i], "itemsInBatch": itemsInBatch[i]}
    batchList.append(batch)
    
#%%
# distances = _demo.distance_ij

#%%
# for i in range(24):
#     print(distances.get(('OutD0',str(i))))
    
#%% CREATE DISTANCE MATRICES
# Pull calculated distances out of Xie's code and reshape it into a distance
# matrix. Each dimension is (amount of items + 2 packing stations)
distMat = np.array(list(distances.values())).reshape(itemAmount + 2, itemAmount + 2)

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
    
#%% 

def FUNC_MinimumDistance(nodesOfPods, packingStation):
    currentBatch = nodesOfPods.copy()
    route = [packingStation] 
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
    totalDist = float(totalDist)
    # print("Final dist:")
    # print(totalDist)
    route.append(packingStation)
    return({"Route":route, "Distance": totalDist})

#%%
fromStation0 = FUNC_MinimumDistance(itemsInOrderList[2], "OutD0")

#%%     
finalOrders = []
count = 0
for itemsInOrder in itemsInOrderList:
    print("----Order " + str(count) +"----")
    fromStation0 = FUNC_MinimumDistance(itemsInOrder, "OutD0")
    print("If start at station 0:")
    print(fromStation0)
    fromStation1 = FUNC_MinimumDistance(itemsInOrder, "OutD1")
    print("If start at station 1:")
    print(fromStation1)
    if fromStation0["Distance"] < fromStation1["Distance"]:
        finalOrders.append({"orderID":count, "itemsInOrder" : itemsInOrder, "fromStation" : "OutD0" , "Distance" : fromStation0["Distance"]})
        print("Choose station 0")
    else: 
        finalOrders.append({"orderID":count, "itemsInOrder" : itemsInOrder, "fromStation" : "OutD1" , "Distance" : fromStation1["Distance"]})
        print("Choose station 1")
    count += 1