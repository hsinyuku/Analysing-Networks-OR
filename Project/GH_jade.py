#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:01:29 2020

@author: ElinaKu
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
    item = warehouseInstance.ItemDescriptions[str(itemID)].__dict__
    itemDesc = item["Color"] + "/" + item["Letter"]
    item["Description"] = itemDesc
    return(item)
    
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
        batchInfo.append({"batchID": batchID, "ordersInBatch": b, "routeInBatch" : route, "distance": dist, "weight" : F_weightOfOrdersInBatch(b)})
        batchID += 1
    
    batchInfo = pd.DataFrame(batchInfo)
    return(batchInfo)

#%% F_greedyHeuristic(batchInfo):
def F_greedyHeuristic(batchFromStation, packingStation):
    
    # Initialize
    greedyBatch = []
    batchFromStationCopy = copy.deepcopy(batchFromStation)
    # Extract orders from the station
    orderToCover = station.get(packingStation)
    
    # Extract the feasible batch information for corresponding station from 
    # batchFromStation object
    batchInfo = [batch['batchInfo'] for batch in batchFromStationCopy if batch['station'] == packingStation]
    batchInfo = batchInfo[0]    
    
    while orderToCover != []:
        print("-----Next Batch-----")      
        batchInfo["numberOfBatchCovered"] = 0
        print("Order to cover: " + str(orderToCover))
        
        # Calculate the amount of order a feasible batch can cover
        value = []
        for i in list(batchInfo.index):
            value = value + [len(set(batchInfo.loc[i,"ordersInBatch"]) & set(orderToCover))]
        batchInfo["numberOfBatchCovered"] = value
        
        # # Preliminary criteria 1 - batch has to cover at least 1 order
        # nextBatch = batchInfo.query("numberOfBatchCovered > 0")
        # print("Preliminary criteria - batch has to cover at least 1 order:")
        # print(nextBatch)
        
        # Preliminary criteria 2 - batch is a subset of orderToCover (to avoid
        # a batch being covered more than once)
        nextBatch = copy.deepcopy(batchInfo)
        for i in nextBatch.index:
            canCover = nextBatch.loc[i].ordersInBatch
            if all([compare in orderToCover for compare in canCover]) == False:
                nextBatch = nextBatch.drop(i)
        print("Prelim: Batch in a subset of orderToCover")
        print(nextBatch)    
        
        # Greedy criteria 1 - subset of batches with minimum distance travelled
        nextBatch = nextBatch.query("distance == distance.min()")
        print("Min dist criteria (greedy):")
        print(nextBatch)
        
        # Greedy criteria 2 - subset of batches with maximum number of orders covered
        nextBatch = nextBatch.query("numberOfBatchCovered == numberOfBatchCovered.max()")
        print("Max cover criteria:")
        print(nextBatch)
        
        # From that subset, randomly pick a batch and append it to the final result
        random = rd.randint(0,len(nextBatch)-1)
        nextBatch = dict(nextBatch.iloc[random,:])
        print("Choose random:")
        print(nextBatch)
        greedyBatch.append(nextBatch)
        
        # Delete already covered orders out of orderToCover list
        orderToCover = list(set(orderToCover) - set(nextBatch["ordersInBatch"]))
        
        # Delete the already chosen batch out of the batch list
        batchToDelete = nextBatch["batchID"]
        batchInfo = batchInfo[batchInfo["batchID"] != batchToDelete]
        
    greedyBatch = pd.DataFrame(greedyBatch)
    return(greedyBatch)

#%% F_greedyHeuristicTry(batchInfo): added by Jade
def F_greedyHeuristic_JadeTest(batchFromStation, packingStation):
    
    # Initialize
    greedyBatch = []
    batchFromStationCopy = copy.deepcopy(batchFromStation)
    # Extract orders from the station
    orderToCover = station.get(packingStation)
    
    # Extract the feasible batch information for corresponding station from 
    # batchFromStation object
    batchInfo = [batch['batchInfo'] for batch in batchFromStationCopy if batch['station'] == packingStation]
    batchInfo = batchInfo[0]  #list becomes a pandas dataframe
    #print("batchInfo is:")
    #print(batchInfo)
    
    while orderToCover != []:
        print("-----Next Batch-----")      
        batchInfo["numberOfBatchCovered"] = 0
        print("Order to cover: " + str(orderToCover))
        
        # Calculate the amount of order a feasible batch can cover
        value = []
        for i in list(batchInfo.index): #for i=0~9
            value = value + [len(set(batchInfo.loc[i,"ordersInBatch"]) & set(orderToCover))]
            #print(len(set(batchInfo.loc[i,"ordersInBatch"])))
            #print(set(orderToCover))  #{0,1,6,7,9} for Out0
            #print([len(set(batchInfo.loc[i,"ordersInBatch"]) & set(orderToCover) )])
            #print(type(value)) #list
            #print(value)
        batchInfo["numberOfBatchCovered"] = value
        #print(batchInfo)
        
        # # Preliminary criteria 1 - batch has to cover at least 1 order
        #nextBatch = batchInfo.query("numberOfBatchCovered > 0") 
        #print("Preliminary criteria - batch has to cover at least 1 order:")
        #print(nextBatch)
        
        # Preliminary criteria 2 - batch is a subset of orderToCover (to avoid
        # a batch being covered more than once)
        nextBatch = copy.deepcopy(batchInfo)
        for i in nextBatch.index:
            canCover = nextBatch.loc[i].ordersInBatch #list
            if all([compare in orderToCover for compare in canCover]) == False:
                nextBatch = nextBatch.drop(i)
        print("Prelim: Batch in a subset of orderToCover")
        print(nextBatch)      
        
        # Greedy criteria 1 - subset of batches with minimum distance travelled
        #print(nextBatch.loc[:,'distance'])
        nextBatch = nextBatch.query("distance == distance.min()")
        print("Min dist criteria (greedy):")
        print(nextBatch)
        
        # Greedy criteria 2 - subset of batches with maximum number of orders covered
        nextBatch = nextBatch.query("numberOfBatchCovered == numberOfBatchCovered.max()")
        print("Max cover criteria:")
        print(nextBatch)
        
         # From that subset, randomly pick a batch and append it to the final result
        random = rd.randint(0,len(nextBatch)-1)
        #print("random is: " +str(random))
        #print(type(nextBatch.iloc[random,:])) #a panda series
        #print(dict(nextBatch.iloc[random,:]))
        nextBatch = dict(nextBatch.iloc[random,:])
        print("Choose random:")
        print(nextBatch)
        greedyBatch.append(nextBatch)
        #print(greedyBatch) #a list
        
        # Delete already covered orders out of orderToCover list
        orderToCover = list(set(orderToCover) - set(nextBatch["ordersInBatch"]))
        #print(orderToCover)
        
        # Delete the already chosen batch out of the batch list
        batchToDelete = nextBatch["batchID"]
        #print("Delete: !!!!!!!!!!")
        print(batchToDelete)
        batchInfo = batchInfo[batchInfo["batchID"] != batchToDelete]
        
    
    greedyBatch = pd.DataFrame(greedyBatch)
    #print(greedyBatch)
    return(greedyBatch)

#greedyStation0_test = F_greedyHeuristic_JadeTest(batchFromStation, packingStation = "OutD0")
  
#%% MAIN SCRIPT
# Source Xie's code (assuming the right working directory)
#from instance_demo import *
    
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

#%%
# Final greedy result for each of the station
greedyStation0 = F_greedyHeuristic(batchFromStation, packingStation = "OutD0")
greedyStation1 = F_greedyHeuristic(batchFromStation, packingStation = "OutD1")

#%% Jade_added for randdom Neibour (algorithm 4 in overleaf)

def F_randomDelSol(batchSolution):
    #batchSolution = F_greedyHeuristicTry(batchFromStation, packingStation)
    print("           ")
    
    print("Original Solution: ")
    print(batchSolution)
    batchList = batchSolution["batchID"] #pd.Series
    #print(batchList)
    print("           ")
    
    n = rd.randint(2,len(batchList)-1) 
    print("Numbers of batches to be removed: "+str(n))
    
    idRemoved= rd.sample(range(0,len(batchList)),n) #list
    #print("The removed id(s) : "+str(idRemoved))
    del(n)
    batchRemoved=[]
    for i in idRemoved: batchRemoved.append(batchList[i])
    print("The removed batchID(s): "+str(batchRemoved))
    
    ordersRemoved=[]
    for i in idRemoved:
        order = batchSolution.iloc[i].ordersInBatch
        ordersRemoved.append(list(order))
    print("The removed order(s): "+str(ordersRemoved))
    print("           ")
    
    for i in batchRemoved: 
        batchSolution = batchSolution[batchSolution["batchID"] != i]
    print("The remaining batches in original solution: ")
    print(batchSolution)
    
    # Algorithm to repair the removed orders with random batchtes
        #batchesNew = F_randomPickBatch(ordersRemoved)
        #batchSolutionNew = batchSolutionLeft + batchesNew
        #return(batchSolutionNew)

    # The following is for demonstration purpose, pretending that there are 2 added orders 
    #batchesNewPretend = pd.DataFrame({'batchID':[30,50],'ordersInBatch':[[30,30],[50]]})                                 
    #batchSolutionPretend = batchSolution[["batchID","ordersInBatch"]].append(batchesNewPretend, ignore_index=True)
    #print("The new batchSolution: ")
    #print(batchSolutionPretend)
    
    #return([ordersRemoved,batchSolutionPretend])
    
    return([batchRemoved,ordersRemoved, batchSolution])

#%%
def F_elementsOfList(removedOrders):
    OrderList = [] 
    for removedBatch in removedOrders:
        for order in removedBatch:
            OrderList.append(order)
    return(OrderList)

#%% 
def F_randomPickBatch(batchFromStation,packingStation,removedOrderList):
    batchList = []
    pickedBatch = []
    pickedOrder = []
    batchFromStationCopy = copy.deepcopy(batchFromStation)
    
    removedOrderList = F_elementsOfList(removedOrderList)
    print("removedOrder(s) : "+str(removedOrderList))
    
    batchInfo = [batch['batchInfo'] for batch in batchFromStationCopy if batch['station'] == packingStation]
    batchInfo = batchInfo[0]  #list becomes a pandas dataframe
    #print("batchInfo is:")
    #print(batchInfo)
    print("               ")
  ############################################################################################################
    print("----- Check all feasible batches for those with removed orders(=find possible neibour) -----") 
    for i in list(batchInfo.index):    
        ordersToCheck = batchInfo.loc[i,"ordersInBatch"] 
        print("For batchID "+str(i)+str(", ordersToCheck is : ")+str(ordersToCheck))
        union = set(list(set(ordersToCheck).union(removedOrderList)))
        diff = union.difference(set(removedOrderList))
        if not diff: 
            print("batchID "+str(i)+str(" has order(s)")+str(ordersToCheck)+
                  str(", so is added into Neibours"))
            batchList=[*batchList,i]
        else: print("batchID "+str(i)+str(" is removed"))
    print("----------- Finish checking all feasible batches -------------") 
    print("All possible neibours: "+str(batchList))
    print("               ")
  ############################################################################################################  
    print("---------- Check in all neibours to build new solution ------------") 
    j=0
    while batchList !=[]: 
    #while i <len(batchList):
    #for i in range(len(batchList)):
        j=j+1
        print("               ")
        print("--- Round "+str(j)+" to pick batches to repair removed orders ---")
        batch = rd.sample(batchList,1) 
        for i in batch: batchInt = int(i)
        print("pickedBatch is: "+str(batch))
    
        order = batchInfo.loc[batchInt,"ordersInBatch"]
        print("pickedOrder is: " +str(order))
        
        pickedBatch += batch 
        pickedOrder += order 
        
        i=0
        while i < len(batchList):
        #for i in range(len(batchList)):
            print("Check batch :"+str(batchList[i]))
            checkOrder = batchInfo.loc[batchList[i],"ordersInBatch"]
            intersect = [value for value in checkOrder if value in order]
            #lst3 = [value for value in lst1 if value in lst2]
            
            if intersect != []:
                print("Batch "+ str(batchList[i])+" has the same order(s) "+str(intersect)+" as in batch "+str(batch))
                print("delete batch "+ str(batchList[i])+" from Neibours")
                batchList.remove(batchList[i])
                print("What is left in Neibours :")
                print(batchList)
                #print(i)
            else: 
                print("Batch "+ str(batchList[i])+" stays in the neibours")
                i+=1
    
        print("*** new solution after round "+str(j)+" : ***")
        print("pickedBatch: "+str(pickedBatch))
        print("pickedOrder: "+str(pickedOrder))
    print("               ")
   
    print("There is nothing in neibour to be picked from.")
    print("---------- Finish building new solution ------------")
   ########################################################################################################### 
    print("               ")
    print("All info of the picked batch(es) to repair deleted batch(es): ")
    
    pickedBatchInfo = pd.DataFrame()
    for i in range(len(pickedBatch)): 
        info = batchInfo[batchInfo["batchID"] == pickedBatch[i]]
        #print(info)
        pickedBatchInfo = pickedBatchInfo.append(info,sort=False)
    print(pickedBatchInfo)
    print("               ")
    
  ############################################################################################################    
    print("---------- Check if new solution repairs all removed orders ------------")
    print("removedOrder(s) : "+str(removedOrderList))
    print("pickedOrder(s)  : "+str(pickedOrder))
    print("    ")
    
    return([pickedBatch, pickedOrder, pickedBatchInfo]) 

#%%
def F_newSol(oriSol, removedBatch, removedOrder, remainSol, pickedBatch, pickedOrder, pickedBatchInfo):    
    print("    ")
    print("---------- Check removed & repaired batches and orders ------------")
    
    print("   ")
    print("Ｏriginal solution: ")
    print(oriSol)
    
    print("   ")
    print("removedBatch: "+str(removedBatch))
    print("removedOrder: "+str(removedOrder))
    
    print("   ")
    print("remain batch(es): ")
    print(remainSol)
    
    print("   ")
    print("pickedBatch: "+str(pickedBatch))
    print("pickedOrder: "+str(pickedOrder))
    
    print("   ")
    print("New solution: ")
    newSol=remainSol.append(pickedBatchInfo)
    print(newSol)
    return("New solution")
    
#%%

# results of delete and repair from random neibour

removedBatch0, removedOrder0, remainSol0  = F_randomDelSol(greedyStation0)
pickedBatch0, pickedOrder0,pickedBatchInfo0 = F_randomPickBatch(batchFromStation, "OutD0",removedOrder0)   
F_newSol(greedyStation0, removedBatch0, removedOrder0, remainSol0, pickedBatch0, pickedOrder0, pickedBatchInfo0)

removedBatch1, removedOrder1, remainSol1  = F_randomDelSol(greedyStation1) 
pickedBatch1, pickedOrder1, pickedBatchInfo1 = F_randomPickBatch(batchFromStation, "OutD1",removedOrder1)    
F_newSol(greedyStation1,removedBatch1, removedOrder1, remainSol1, pickedBatch1, pickedOrder1, pickedBatchInfo1)   
    
    
    
    
    
    


