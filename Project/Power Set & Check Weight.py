# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:54:23 2020

@author: 93cha
"""


import xml.etree.ElementTree as ET
import pandas as pd
import copy

#%% Extract the item descriptions from XML file to a dictionary
    
    # Parse XML file to tree element
    tree = ET.parse("data/sku24/orders_20_mean_5_sku_24.xml")
    
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

    print(itemList)    

#%% Extract the order descriptions from XML file to a dictionary
    orderList = list()
   
    for i in range(len(root[2])):
        orderList.append(list())
        for j in range(len(root[2][i][0])):
            orderList[i].append(root[2][i][0][j].attrib)
            
#%% Fill the weight information of each item to the orderList
    for i in range(len(orderList)):
        for j in range(len(orderList[i])):
            itemID = orderList[i][j]["ItemDescriptionID"]
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
   
orderSet = list(range(0,20))
feasibleBatch = list()
feasibleBatch.append([orderSet.pop(0)])
feasibleBatchWithWeight = list()
    
#%%
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
        FUNC_batchWeight(batch, orderWeight)
        
#%%
        feasibleBatchWeight = list()
        for batch in feasibleBatch:
            if (FUNC_batchWeight(batch, orderWeight) < 10):
                feasibleBatchWeight.append(batch)