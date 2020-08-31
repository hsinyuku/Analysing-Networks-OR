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
import numpy.random as rn


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

  
#%% MAIN SCRIPT
# Source Xie's code (assuming the right working directory)
from instance_demo_Jade import *
    
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

#%% ##### Start from here is added by Jadefor SAA ######

greedyStation0=greedyStation0.drop(["numberOfBatchCovered"],axis=1)
greedyStation1=greedyStation1.drop(["numberOfBatchCovered"],axis=1)

#%%
def F_randomDelSol(batchSolution):
    #batchSolution = F_greedyHeuristic(batchFromStation, packingStation)
    print("           ")
    
    print("Original Solution: ")
    print(batchSolution)
    batchList = batchSolution["batchID"] #pd.Series
    #print(batchList)
    print("           ")
    
    #n=1
    n = rd.randint(1,max(len(batchList)-1,1)) 
    print("Numbers of batches to be removed: "+str(n))
    
    idRemoved= rd.sample(range(0,len(batchList)),n) #list
 
    #print("The removed id(s) : "+str(idRemoved))
    del(n)
    #print("WWW")
    batchRemoved=[]
    #print(batchList)
    for i in idRemoved: batchRemoved.append(batchList[i])
    #print("The removed batchID(s): "+str(batchRemoved))
    #print("YYY")
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
    #print("removedOrder(s) : "+str(removedOrderList))
    
    batchInfo = [batch['batchInfo'] for batch in batchFromStationCopy if batch['station'] == packingStation]
    batchInfo = batchInfo[0]  #list becomes a pandas dataframe
    #print("batchInfo is:")
    print("               ")
  ############################################################################################################
    print("----- Check all feasible batches for those with removed orders(=find possible neibour) -----") 
    for i in list(batchInfo.index):    
        ordersToCheck = batchInfo.loc[i,"ordersInBatch"] 
        #print("For batchID "+str(i)+str(", ordersToCheck is : ")+str(ordersToCheck))
        union = set(list(set(ordersToCheck).union(removedOrderList)))
        diff = union.difference(set(removedOrderList))
        if not diff: 
            #print("batchID "+str(i)+str(" has order(s)")+str(ordersToCheck)+
                 # str(", so is added into Neibours"))
            batchList=[*batchList,i]
        #else: print("batchID "+str(i)+str(" is removed"))
    #print("----------- Finish checking all feasible batches -------------") 
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
            #print("Check batch :"+str(batchList[i]))
            checkOrder = batchInfo.loc[batchList[i],"ordersInBatch"]
            intersect = [value for value in checkOrder if value in order]
            #lst3 = [value for value in lst1 if value in lst2]
            
            if intersect != []:
                #print("Batch "+ str(batchList[i])+" has the same order(s) "+str(intersect)+" as in batch "+str(batch))
                #print("delete batch "+ str(batchList[i])+" from Neibours")
                batchList.remove(batchList[i])
                #print("What is left in Neibours :")
                print(batchList)
                #print(i)
            else: 
                #print("Batch "+ str(batchList[i])+" stays in the neibours")
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
        #pickedBatchInfo.drop(columns=["numberOfBatchCovered"])
    print(pickedBatchInfo)
    print("               ")
    
  ############################################################################################################    
    print("---------- Check if new solution repairs all removed orders ------------")
    print("removedOrder(s) : "+str(removedOrderList))
    print("pickedOrder(s)  : "+str(pickedOrder))
    print("    ")
    
    return([pickedBatch, pickedOrder, pickedBatchInfo]) 

    
#%%
def F_randomNeighbour(oriSol, Station):
    
    #oriSolcopy=copy.deepcopy(oriSol)
    #oriSol=oriSolcopy.drop(["numberOfBatchCovered"],axis=1)
    #print("AAAAA")
    #print(oriSol)
    
    removedBatch, removedOrder, remainSol  = F_randomDelSol(oriSol)   
    #remainSol=remainSol.drop(["numberOfBatchCovered"],axis=1)
    #print("BBBBBb")
    #print(remainSol)
    
    repairSol = (F_randomPickBatch(batchFromStation, Station, removedOrder))[2]
    #print("CCC")
    #print(repairSol)
    #print("DDDD")
    
    newSol=remainSol.append(repairSol)
    
    print("   ")
    print("Original solution: ")
    print(oriSol)
    print("   ")
    print("Rabdom neibour of original solution: ")
    print(newSol)
    
    oriDis = sum(oriSol["distance"])
    print("   ")
    print("original total distance: "+str(oriDis))
    newDis = sum(newSol["distance"])
    print("new total distance:      "+str(newDis))
    
    return(oriDis, newSol, newDis)   

#Neighbour1 = F_randomNeighbour(greedyStation1, "OutD1")
#Neighbour0 = F_randomNeighbour(greedyStation0, "OutD0")

#%%
def accept_prob(cost, new_cost, T):
    if new_cost < cost: return 1
    else: 
        p = np.exp(- (new_cost - cost) / T)
        return p
       
#%%
def SAA_old(oriSol, Station, T, alpha, tempLimit):
    print("  ")
    print("------------------ For station "+str(Station) +"------------------")
    T=T
    alpha = alpha
    epsilon = tempLimit
    
    optSol=pd.DataFrame()
    optSol = optSol.append(oriSol) 
    print("optSol = oriSol")
    
    optDis=sum(optSol["distance"])
    print("optDis is the oriDis: "+str(optDis))

    DisRec=[]
    DisRec.append(optDis)
    print("DisRec: "+str(DisRec))
    
    optDisRec=[]
    optDisRec.append(optDis)
    print("optDisRec: "+str(optDisRec))
    
    #if T==1000:    
    round=0
    while T > epsilon:  
      round=round+1
      print("   ")
      print(" ************ Round "+str(round)+": T = "+str(T)+" ***************")
      #print("GGGGGGGGGGGGGGGGGGGGGGGGG")
      oriSol=oriSol.reset_index(drop=True)
      #print("GGGGGGGGGGGGGGGGGGGGGGGGG")
      neighborInfo = F_randomNeighbour(oriSol, Station)
      newSol = neighborInfo[1]
      oriDis = neighborInfo[0]
      newDis = neighborInfo[2]
      
      #Accept the neibour if it has smaller distance
      p = accept_prob(oriDis, newDis, T)
      uRan = rn.random()
      #print("AAAAAAAAAAAAAAAAAAAAAAA")
      print("   ")
      print("p= "+str(p))
      print("uRan= "+str(uRan))
      #print("AAAAAAAAAAAAAAAAAAAAAAA")
      #if newDis < oriDis:
      if p > uRan:
          print("P > random U(0,1) --> accept newSol")
          oriSol = copy.deepcopy(newSol)
          
      #  reject 
      else: 
          print("Reject new solution")
      
      if newDis < optDis:
          optSol = copy.deepcopy(newSol)
          optDis = sum(optSol["distance"])
          optDisRec.append(optDis)
      
      DisRec.append(newDis)
      print("   ")
      print("DisRec: "+str(DisRec))  
      print("optDisRec: "+str(optDisRec))
      print("Optimal distance at round "+str(round)+" is "+str(optDisRec[-1]))
          
      T = alpha*T
    
    return(optDis, DisRec, optDisRec)

#%%
def SAA_pertubation(oriSol, Station, T, alpha, tempLimit, beta, iteration):
    for i in range(iteration):
        T = beta*T 
        print("   ")
        print("################# Iteration "+str(i+1)+" starts: #################")
        SAA(oriSol,Station, T, alpha, tempLimit)
        print("   ")
        print("Iteration "+str(i+1)+" ended")
        

SAA_pertubation(greedyStation1,"OutD1", T=10/0.8, alpha=0.8, tempLimit=7, beta=0.8, iteration=2) 


#%% Executing results of SAA

optDis1 = SAA_old(greedyStation1,"OutD1", T=10, alpha=0.8, tempLimit=5)

#%%
optDis0 = SAA_old(greedyStation0,"OutD0", T=10, alpha=0.8, tempLimit=5)
print("  ")
print("Optimal distance of all stations: "+str(optDis1+optDis0))

#%% End of the file to be executed - Jade


def F_randomPickBatch(batchFromStation,packingStation,removedOrderList):
    batchList = []
    pickedBatch = []
    pickedOrder = []
    batchFromStationCopy = copy.deepcopy(batchFromStation)
    
    removedOrderList = F_elementsOfList(removedOrderList)
    print("removedOrder(s) : "+str(removedOrderList))
    
    return(removedOrderList) 
    
pickedBatch0, pickedOrder0,pickedBatchInfo0 = F_randomPickBatch(batchFromStation, "OutD0",removedOrder0)