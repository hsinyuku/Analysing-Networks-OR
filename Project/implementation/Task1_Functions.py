# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:54:23 2020

@author: 93cha
"""

# Functions rely on the script instance_demo.py to be sources, as this script 
# generates the object warehouseInstance that contains all relevant information
# about the project

#%% import libraries
import xml.etree.ElementTree as ET
import pandas as pd
import copy
import numpy as np
import random as rd

#%% F_itemInfo
def F_itemInfo(itemID):
    """ Return a dictionary with information about the item specified by itemID.
    Function relies on warehouseInstance to be loaded.

    Parameters
    ----------
    itemID : int

    Returns
    -------
    Dictionary contains ID, Type, Weight, Letter, Color and Description.

    """
    item = warehouseInstance.ItemDescriptions[str(itemID)].__dict__
    itemDesc = item["Color"] + "/" + item["Letter"]
    item["Description"] = itemDesc
    return(item)
    
#%% F_orderInfo
def F_orderInfo(orderID):
    """Returns a list of dictionaries with information about an order specified
    by orderID. Function relies on warehouseInstance to be loaded.

    Parameters
    ----------
    orderID : int

    Returns
    -------
    List contains one dictionary for each itemID in the specified order, with
    respective weight and count of items for each itemID.
    """
    itemList = []
    for itemID in warehouseInstance.Orders[orderID].Positions.keys():
        weight = F_itemInfo(itemID)["Weight"]
        count = int(warehouseInstance.Orders[orderID].Positions[str(itemID)].Count)
        item = {"itemID": int(itemID), "Weight" : weight, "Count" : count}
        itemList.append(item)
    return(itemList)

#%% F_itemsInOrder
def F_itemsInOrder(orderID):
    """For a specified orderID, returns a list of integers. Each integer is an
    item in the order.
    POTENTIALLY A DUPLICATE

    Parameters
    ----------
    orderID : int

    Returns
    -------
    List of integers.

    """
    items = [*warehouseInstance.Orders[orderID].Positions]
    items = [int(i) for i in items]
    return(items)

#%% F_weightOfOrder
def F_weightOfOrder(orderID):
    """For a given orderID, calculates the total weight of all items in the
    order.

    Parameters
    ----------
    orderID : int

    Returns
    -------
    Float.

    """
    return(round(sum([dict["Weight"]*dict["Count"] for dict in F_orderInfo(orderID)]), 2))

#%% F_weightOfBatch
def F_weightOfOrdersInBatch(orderList):
    """
    For a given list of orders (as orderIDs), calculates the total weight of 
    all items in all orders in the list.

    Parameters
    ----------
    orderList : list of int

    Returns
    -------
    float

    """
    
    #totalWeight = 0
    #for order in orderList:
    #    totalWeight += F_weightOfOrder(order)
    # return(totalWeight)
    return(sum([F_weightOfOrder(orderID) for orderID in orderList]))

#%%
def F_createDistMat():
    """For each station in the warehouse, creates one DataFrame in the global
    environment that contains the distance from the station to each pod and 
    all other stations.
    
    Requires
    --------
    distance_ij from the script instance_demo.py must have been loaded.

    Returns
    -------
    None; instead directly writes in the global environment.

    """
    global distToStation0
    global distToStation1
    global distMat

    # What does this do?
    stopAmount = int(np.sqrt(len(distance_ij)))
        # StopAmount is the number of elements in the matrix (number of rows)
    distMat = np.array(list(distance_ij.values())).reshape(stopAmount, stopAmount)
        # reshape distance_ij from a 1d-array to a 2d-matrix
    
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

#%% LUKAS: understood until here
    

#%% 
def F_itemsInBatch(batchList): 
    """???

    Parameters
    ----------
    batchList : List of 

    Returns
    -------
    ???

    """
    itemList = []
    for orderID in batchList:
        items = F_itemsInOrder(orderID)
        itemList = list(set(itemList) | set(items))
    return(itemList)

#%% F_minDistance
    # Comment from Lukas: did not yet fully check the function, but I did 
    # understand its basic functionality.
def F_minDist(items, packingStation):
    """For a given list of items and a packing station, returns the shortest
    possible route of pods and stations, where each item is contained in one 
    of the pods.

    Parameters
    ----------
    items : List of int; itemIDs in a list, usually within one order.
    packingStation : String. Stations names have the form of OutDX, where X is
    an integer representing the station number.

    Returns
    -------
    None.

    """
    # Call for distance matrix
    F_createDistMat()    
    
    # Create destructive copy of nodesOfPods, so that each
    # node can be removed later
    currentBatch = items.copy()
    
    # First node of route is the chosen packing station
    route = [packingStation] 
    
    # Initialize distance
    totalDist = 0

    for i in range(len(items)):
        
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
    # Comment from Lukas: did not yet fully check the function, but I did 
    # understand its basic functionality.
def F_assignOrderToStation(orderAmount, type = "full"):
    """ ???
    
    Parameters
    ----------
    orderAmount : int
        Number of orders, has to be specified externally. 
    type : TYPE, optional
        DESCRIPTION. The default is "full".

    Returns
    -------
    None.

    """
    
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
    """ 

    Parameters
    ----------
    listOfOrders : TYPE
        DESCRIPTION.
    packingStation : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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


#%% Jade_added (All above this section is written by Chan)

def F_randomDelSol(batchSolution):
    """Deletes random batches from a given solution.

    Parameters
    ----------
    batchSolution : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
    """Extract orders from batches into one single list.

    Parameters
    ----------
    removedOrders : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    OrderList = [] 
    for removedBatch in removedOrders:
        for order in removedBatch:
            OrderList.append(order)
    return(OrderList)

#%% 
def F_randomPickBatch(batchFromStation,packingStation,removedOrderList):
    """After deleting batches, pick from feasible batches to repair the
    removed orders.

    Parameters
    ----------
    batchFromStation : TYPE
        DESCRIPTION.
    packingStation : TYPE
        DESCRIPTION.
    removedOrderList : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
   
    print("There is nothing in neighbour to be picked from.")
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
    """
    Combine the original batch, from which some batches have been removed, 
    with the repaired batches, to make a new solution.

    Parameters
    ----------
    oriSol : TYPE
        DESCRIPTION.
    removedBatch : TYPE
        DESCRIPTION.
    removedOrder : TYPE
        DESCRIPTION.
    remainSol : TYPE
        DESCRIPTION.
    pickedBatch : TYPE
        DESCRIPTION.
    pickedOrder : TYPE
        DESCRIPTION.
    pickedBatchInfo : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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

