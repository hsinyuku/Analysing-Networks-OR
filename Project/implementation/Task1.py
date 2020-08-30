# The script is divided into three main parts:
# A) importing files and creating instances defined in instance_demo.py (this is the code we were provided with)
# B) functions that contain the algorithms to solve the problems
# C) execution of the given functions

#%% sourcing scripts
import rafs_instance
import instance_demo as demo
import pandas as pd
import copy
import numpy as np
import random as rd

#%% control knobs

# SPecify mean item amount in an order (either 1x6 or 5)
meanItemInOrder = "1x6"

# Specify the number of orders we receive (either 10 or 20)
orderAmount = 10

# Specify the number of items in the warehouse (either 24 or 360)
itemAmount = 24

# Specify the policy (either "dedicated_1" or "mixed_shevels_1-5")
podPolicy = "dedicated_1"

test = 'data/sku''/pods_infos.txt'

#%% A) importing files and creating instances defined in instance_demo.py

# not sure what this contains, but it is needed for the Warehouse-class
layoutFile = r'data/layout/1-1-1-2-1.xlayo' 
# loading all the information about the pods
podInfoFile = 'data/sku' + str(itemAmount) + '/pods_infos.txt'   
# loading information about picking locations, packing stations, waypoints,
# pods 
instanceFile = r'data/sku' + str(itemAmount) + '/layout_sku_' + str(itemAmount) + '_2.xml'

# loading information about item storage: contains all SKUs along with their
# attributes
storagePolicyFile = 'data/sku' + str(itemAmount) + '/pods_items_' + str(podPolicy) + '.txt'
#storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

# loading information about the orders: contains list of orders, with number
# of ordered items per SKU-ID
orderFile =r'data/sku' + str(itemAmount) + '/orders_' + str(orderAmount) + '_mean_' + str(meanItemInOrder) + '_sku_' + str(itemAmount) + '.xml'
#orders['20_5']=r'data/sku24/orders_20_mean_5_sku_24.xml'

# trying a different way to get the demonstration running
# function to prepare data
warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile, orderFile)

batch_weight = 18
item_id_pod_id_dict = {}

distance_ij = demo.WarehouseDateProcessing(warehouseInstance).CalculateDistance()


#%% Necessary functions

# Functions rely on the script instance_demo.py to be sources, as this script 
# generates the object warehouseInstance that contains all relevant information
# about the project



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
    """ For a given sset of orders, returns a list of all feasible batches that
    can be generated using these orders.

    Parameters
    ----------
    listOfOrders : list
        List of orders, represented as a list of integers.
    packingStation : string
        String that specifies which station the orders belong to.

    Returns
    -------
    A DataFrame with the columns batchID (integer identifying a batch), 
    ordersInBatch (list of integers representing orders that are in the batch),
    routeInBatch (list of integers representing the pods that are visited 
    in the batch), distance (float representing distance travelled in the
    batch) and weight (float, cumulative weight of all items in the batch).
    """
    batch = [] # initialise list of feasible batches as empty list
    batch.append([listOfOrders.pop(0)]) # add the first order to the batch,
        # remove the first order from the list of orders
    
    while (listOfOrders != []):
        # make a copy of the list of batches
        batchCopy = copy.deepcopy(batch) 
        
        # pick the next order from the list orders, remove this order from the 
        # list of orders
        nextOrder = [listOfOrders.pop(0)]
        
        # Add nextOrder to each of the batches in batchCopy to create new batches
        batchCopy = [b + nextOrder for b in batchCopy]
        
        # Check weight capacity constraint of newly created batches
        # batch_weight is created by instance_demo and a global variable
        batchCopy = [b for b in batchCopy if F_weightOfOrdersInBatch(b) <= batch_weight]
        
        # Add nextOrder to batch as a feasible batch
        batch.append(nextOrder)
        
        # Append new feasible batches in batchCopy to batch
        batch = batch + batchCopy
    
    # initialise a new list to store data about the batches
    batchInfo = []    
    batchID = 0
    # for each batch, generate a batchID, extract the orders in the batch, the 
    # route taken to travel the batch, the distance travelled, and the total 
    # weight of all items in the batch
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
    """Greedy heuristic to assign orders to batches, for an already given
    assignment of orders to stations, and for only one station.

    Parameters
    ----------
    batchFromStation : list of dictionaries
        For each station, the list contains one dictionary. Each dictionary
        contains two values: the first is the name of the station, the
        second is a DataFrame of feasible batches as generated by the function
        F_orderToBatch().
        
    packingStation : String
        String that specifies for each station to apply the heuristic.

    Returns
    -------
    DataFrame, containing six columns: batchID (IDs of batches used to cover the
    orders), ordersInBatch (for each batch, which orders it contains),
    routeInBatch (for each batch, which pods it travels to), distance (for each
    batch, the distance necessary to fulfil the batch), weight (weight of all
    items in that batch) and numberOfBatchCovered (number of orders covered by
    the batch)

    """
    # Initialize empty lists
    greedyBatch = []
    batchFromStationCopy = copy.deepcopy(batchFromStation)
    
    # Extract orders from the station: orderToCover is a list of all orders
    # of a station, that is, the set of all orders that need to be covered;
    # station is a dictionary that has one key for each station and as value
    # a list of all orderIDs that are assigned to the station.
    orderToCover = station[packingStation]
    
    # Extract the feasible batch information for the respective station from 
    # batchFromStation object
    batchInfo = [batch['batchInfo'] for batch in batchFromStationCopy if batch['station'] == packingStation]
    # This is somehow necessary because to this point, batchInfo is a list 
    # that contains only one element, which is the dataframe we need
    batchInfo = batchInfo[0]
    
    while orderToCover != []:
        print("-----Next Batch-----")
        # add a new column called numberOfBatchCovered (will be filled later        )
        batchInfo["numberOfBatchCovered"] = 0
        print("Order to cover: " + str(orderToCover))
        
        # For each batch, calculate the number of orders a feasible batch can
        # cover. This number is equal to the orders that are in both a batch
        # and the set of orders that still need to be covered.
        # For each row in the DataFrame that contains the data about feasible
        # batches, count the number of orders that are in both this feasible
        # batch and in the cover
        value = []
        for i in list(batchInfo.index):
            value = value + [len(set(batchInfo.loc[i,"ordersInBatch"]) & set(orderToCover))]
        batchInfo["numberOfBatchCovered"] = value
        
        # # Preliminary criterion 1 - batch has to cover at least 1 order
        # nextBatch = batchInfo.query("numberOfBatchCovered > 0")
        # print("Preliminary criteria - batch has to cover at least 1 order:")
        # print(nextBatch)
        
        # Preliminary criterion 2 - batch is a subset of orderToCover (to avoid
        # a batch being covered more than once)
        nextBatch = copy.deepcopy(batchInfo)
        for i in nextBatch.index:
            canCover = nextBatch.loc[i].ordersInBatch
            if all([compare in orderToCover for compare in canCover]) == False:
                nextBatch = nextBatch.drop(i)
        print("Prelim: Batch in a subset of orderToCover")
        print(nextBatch)    
        
        # Greedy criteria 1 - subset of batches with maximum number of orders covered
        nextBatch = nextBatch.query("numberOfBatchCovered == numberOfBatchCovered.max()")
        print("Max cover criteria:")
        print(nextBatch)
        
        # Greedy criteria 2 - subset of batches with minimum distance travelled
        nextBatch = nextBatch.query("distance == distance.min()")
        print("Min dist criteria (greedy):")
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


#%% 

def F_randomDelSol(batchSolution):
    """Deletes a random number of random batches from a given solution. Works
    on one station at a time.

    Parameters
    ----------
    batchSolution : DataFrame, returned from function F_greedyHeuristic
        contains six columns: batchID (IDs of batches used to cover the orders),
        ordersInBatch (for each batch, which orders it contains), routeInBatch
        (for each batch, which pods it travels to), distance (for each batch,
        the distance necessary to fulfil the batch), weight (weight of all items
        in that batch) and numberOfBatchCovered (number of orders covered by
        the batch)

    Returns
    -------
    A list with three elements. First element is a list of integers, containing
    the batch IDs that have been removed. Second element is a list, with length
    equal to the number of batches removed, and a list of integers for each
    batch removed, which represent the orders that are in the removed batches.
    Third element is a DataFrame, which contains the "rest" of the solution, 
    that is all the information about the batches not removed from the solution.

    """
    #batchSolution = F_greedyHeuristicTry(batchFromStation, packingStation)
    print("           ")
    
    print("Original Solution: ")
    print(batchSolution)
    batchList = batchSolution["batchID"] #pd.Series
    #print(batchList)
    print("           ")
    
    # draw a random number to determine how many batches to delete from the solution
    n = rd.randint(2,len(batchList)-1) 
    print("Numbers of batches to be removed: "+str(n))
    
    # determine which batches to delete from the solution
    idRemoved= rd.sample(range(0,len(batchList)),n) #list
    del(n)
    # copy batches to be deleted from the solution
    batchRemoved=[]
    for i in idRemoved: batchRemoved.append(batchList[i])
    print("The removed batchID(s): "+str(batchRemoved))
    
    # get list of orders that are in the removed batches
    ordersRemoved=[]
    for i in idRemoved:
        order = batchSolution.iloc[i].ordersInBatch
        ordersRemoved.append(list(order))
    print("The removed order(s): "+str(ordersRemoved))
    print("           ")
    
    # delete batches from the solution
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
    removedOrders : 
        List of lists of integers. Second output from F_randomDelSol().
        Every integer is an orderID, and every list groups together orderIDs 
        that are in a batch.

    Returns
    -------
    List of integers, representing orderIDs that are in list of lists provided.

    """
    OrderList = [] 
    for removedBatch in removedOrders:
        for order in removedBatch:
            OrderList.append(order)
    return(OrderList)

#%% 
def F_randomPickBatch(batchFromStation, packingStation, removedOrderList):
    """After deleting batches, pick from feasible batches to repair the
    removed orders.

    Parameters
    ----------
    batchFromStation : list of dictionaries
        List with one element per station, which contains a dictionary. See also
        the help for F_greedyHeuristic(), which uses the same object.
    packingStation : String
        Specifies the packing station on which to operate.
    removedOrderList : first output from F_randomDelSol()
        list of integers, containing the batch-IDs that have been removed from 
        a previous solution.

    Returns
    -------
    ???

    """
    
    # initialise empty lists
    batchList = []
    pickedBatch = []
    pickedOrder = []
    batchFromStationCopy = copy.deepcopy(batchFromStation)
    
    # extract orders from the batches that were removed
    removedOrderList = F_elementsOfList(removedOrderList)
    print("removedOrder(s) : "+str(removedOrderList))
    
    # extract the dataframe with information about all feasible batches
    batchInfo = [batch['batchInfo'] for batch in batchFromStationCopy if batch['station'] == packingStation]
    batchInfo = batchInfo[0]  #list becomes a pandas dataframe
    
    # for each feasible batch, check whether it only contains orders that have 
    # been removed in the previous breaking of the solution. 
    print("")
    print("For all feasible batches, check which can be used to repair the solution.") 
    print("-------------------------------------------------------------------------")
    for i in list(batchInfo.index):
        ordersToCheck = batchInfo.loc[i,"ordersInBatch"] 
        print("For batchID "+str(i)+str(", ordersToCheck is : ")+str(ordersToCheck))
        # set of orders that are in the current batch and among the removed orders
        union = set(list(set(ordersToCheck).union(removedOrderList)))
        # elements that are in union, but not in removedOrderList
        diff = union.difference(set(removedOrderList))
        # if the batch only contains orders that have been removed in the previous
        # breaking of the solution, it is added to the list of batches from which
        # the solution can be repaired
        if not diff: 
            print("batchID "+str(i)+str(" has order(s)")+str(ordersToCheck)+
                  str(", so is added into neighbours"))
            batchList=[*batchList,i]
        else: print("batchID "+str(i)+str(" is removed"))
        
    print("Finished checking all feasible batches.")
    print("All possible neighbours: " + str(batchList))
    print("")
    
    # 
    print("Check in all neighbours to build new solution")
    print("---------------------------------------------")
    j=0
    while batchList !=[]: 
        j=j+1
        print("")
        print("Round " + str(j) + " to pick batches to repair removed orders")
        # select a random batch
        batch = rd.sample(batchList , 1)
        # not sure what that does
        for i in batch: batchInt = int(i)
        print("pickedBatch is: " + str(batch))
        # extract the orders that are in the randomly selected batch
        order = batchInfo.loc[batchInt,"ordersInBatch"]
        print("pickedOrder is: " + str(order))
        
        # add the randomly selected batch and its orders to the list to keep track
        pickedBatch += batch 
        pickedOrder += order 
        
        # remove ord
        i=0
        while i < len(batchList):
            print("Check batch :" + str(batchList[i]))
            checkOrder = batchInfo.loc[batchList[i],"ordersInBatch"]
            intersect = [value for value in checkOrder if value in order]
            if intersect != []:
                print("Batch "+ str(batchList[i]) + " has the same order(s) " + str(intersect) + " as in batch "+str(batch))
                print("delete batch "+ str(batchList[i])+" from neighbours")
                batchList.remove(batchList[i])
                print("What is left in neighbours :")
                print(batchList)
                #print(i)
            else: 
                print("Batch "+ str(batchList[i])+" stays in the neighbours")
                i+=1
    
        print("*** new solution after round " + str(j) + " : ***")
        print("pickedBatch: "+str(pickedBatch))
        print("pickedOrder: "+str(pickedOrder))
    print("               ")
   
    print("There is nothing in neighbour to be picked from.")
    print("---------- Finish building new solution ------------")
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
    with the repaired batches, to make a new solution. This function prints out
    the solutions already calculated, but does not find solutions.
    
    Parameters
    ----------
    oriSol : DataFrame
        output from F_greedyHeuristic()
    removedBatch, removedOrder, remainSol :
        output from F_randomDelSol()
    pickedBatch,  pickedOrder, pickedBatchInfo : TYPE
        output from F_randomPickBatch()

    Returns
    -------
    None.

    """
    print("    ")
    print("---------- Check removed & repaired batches and orders ------------")
    
    print("   ")
    print("Original solution: ")
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
    

#%% Task 1
# From all orders, assign them to the optimal packing station:
# I use option "lite" here to get two list of orders for two packing station, as
# input for the next function. To see information of the orders, use option "full"
station = F_assignOrderToStation(orderAmount, "lite")

# For each station, create feasible batches from the orders, which also includes
# the sequence of pods visited for the cobot, as well as the minimum distance
# travelled for that batch.
batchFromStation = []

for i in range(len(station)):
    stationCopy = copy.deepcopy(station)
        # calls the name of the packing station
    packingStation = list(stationCopy.keys())[i]
        # calls the orders that belong to that station
    listOfOrders = list(stationCopy.values())[i] 
        # assign orders from list of orders to batches, procuding all feasible
        # batches for each station
    batchFromStation.append({"station":packingStation, "batchInfo":F_orderToBatch(listOfOrders, packingStation)})
del stationCopy

# Final result for each of the station: for each station, take all feasible 
# batches and run the greedy heuristic to choose the optimal sequence of 
# batches
greedyStation0 = F_greedyHeuristic(batchFromStation, packingStation = "OutD0")
greedyStation1 = F_greedyHeuristic(batchFromStation, packingStation = "OutD1")

#%% Task 2
# results of delete and repair from random neibour

removedBatch0, removedOrder0, remainSol0  = F_randomDelSol(greedyStation0)
pickedBatch0, pickedOrder0,pickedBatchInfo0 = F_randomPickBatch(batchFromStation, "OutD0",removedOrder0)
F_newSol(greedyStation0, removedBatch0, removedOrder0, remainSol0, pickedBatch0, pickedOrder0, pickedBatchInfo0)

removedBatch1, removedOrder1, remainSol1  = F_randomDelSol(greedyStation1) 
pickedBatch1, pickedOrder1, pickedBatchInfo1 = F_randomPickBatch(batchFromStation, "OutD1",removedOrder1)
F_newSol(greedyStation1,removedBatch1, removedOrder1, remainSol1, pickedBatch1, pickedOrder1, pickedBatchInfo1)

#%% Task 3: Extension
# 1. Assuming that the items of an SKU can be stored in different shelves and different SKUs can stored within a shelf. We call this storage policy as mixed policy. Extend your implemented heuristics in the previous subsections to support this assumption.

#%% importing files for mixed shelf policy and creating the instances needed

layoutFile = r'data/layout/1-1-1-2-1.xlayo' 
# loading all the information about the pods
podInfoFile = r'data/sku24/pods_infos.txt'
# loading information about picking locations, packing stations, waypoints,
# pods 
instanceFile = r'data/sku24/layout_sku_24_2.xml'

# loading information about item storage: contains all SKUs along with their
# attributes
storagePolicyFile = r'data/pods/pods_items_mixed_shevels_1-5_24.txt'
#storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

# loading information about the orders: contains list of orders, with number
# of ordered items per SKU-ID
orderFile =r'data/sku24/orders_10_mean_5_sku_24.xml'
#orders['20_5']=r'data/sku24/orders_20_mean_5_sku_24.xml'

# trying a different way to get the demonstration running
# function to prepare data
warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile, orderFile)

batch_weight = 18
item_id_pod_id_dict = {}

distance_ij = demo.WarehouseDateProcessing(warehouseInstance).CalculateDistance()

#%% getting data

# getting data about which items are in which pods, building a dictionary that is easier to read
podInfoDict = {int(pod):[{"Description":item.ID, "Count":int(item.Count)} for item in warehouseInstance.Pods[pod].Items] for pod in warehouseInstance.Pods}

# getting data about the items
itemsInfoDict = [{"ID":int(warehouseInstance.ItemDescriptions[SKU].ID), "Description":warehouseInstance.ItemDescriptions[SKU].Color + "/" + warehouseInstance.ItemDescriptions[SKU].Letter, "Weight":warehouseInstance.ItemDescriptions[SKU].Weight} for SKU in warehouseInstance.ItemDescriptions]

# getting data about orders
orderInfoDict = {order:[{"itemID":int(position), "quantity":int(warehouseInstance.Orders[order].Positions[position].Count)} for position in warehouseInstance.Orders[order].Positions] for order in range(0, len(warehouseInstance.Orders))}

# getting data about packing stations
packingStationNames = list(warehouseInstance.OutputStations.keys())

#%% replacing item description in podInfoDict with item ID
# getting an item ID for an item description
for pod in podInfoDict.keys():
    for podItem in podInfoDict[pod]:
        podItem["ID"] = [item["ID"] for item in itemsInfoDict if item["Description"] == podItem["Description"]][0]


#%% converting data to DataFrames, where it helps
for pod in podInfoDict:
    podInfoDict[pod] = pd.DataFrame(podInfoDict[pod]).set_index("ID")
#%%

for order in orderInfoDict:
    orderInfoDict[order] = pd.DataFrame(orderInfoDict[order]).set_index("itemID")
    
itemsInfoDict = pd.DataFrame(itemsInfoDict).set_index("ID")
#%%
# -----------------------------------------------------
# greedy heuristic to determine the route with which to fulfil an order, and its fitness value
# pick an order
order = copy.deepcopy(orderInfoDict[0])
# pick a station
station = packingStationNames[0]
podCopy = copy.deepcopy(podInfoDict)

def F_fitnessValueOrderToStation(order, station, podCopy):
    """For a given station and order, returns the fitness value of that order. Currently, the fitness order is the distance travelled to fulfil the order.
    The distance is calculated greedily: for all items in the order, the algorithm looks for the nearest pod that contains any of the items and chooses to visit it. From this pod, it tries to pick as many items as possible.

    Parameters
    ----------
    order : 
        typically a deep copy of .orderInfoDict
    station : TYPE
        one of packingStationNames
    podInfo : TYPE
        typically a deep copy of podInfoDict.

    Returns
    -------
    None.

    """
    # get a path
    route = [station]
    routeInfo = {}
    # set the distance of the path to 0
    distance = {}
    # iterate over the order and go to the next pod as long as there are items left 
    # in the order
    while order.shape[0] != 0:
        # find the nearest pod that contains at least one item of the SKU
        possiblePodsDistances = []
        # for each SKU, find the nearest pod that contains at least one item of that SKU
        for SKU in order.index:
            # find all pods that contain at least one item of the SKU and get the minimum
            # distance to any of them
            #print("---- SKU ----")
            #print(SKU)
            SKUNearestPod = {}
            for podID, items in podCopy.items():
                if SKU in items.index:
                    #print("-- podID: " + str(podID) +" --")
                    #print(items)
                    #print("--distance: " + str(round(distance_ij[(str(route[-1]), str(podID))], 1)))
                    SKUNearestPod[podID] = round(distance_ij[(str(route[-1]), str(podID))], 1)
            #print(min(SKUNearestPod))
            #print(SKUNearestPod[min(SKUNearestPod)])
            possiblePodsDistances.append({"itemID":SKU,
                                          "pod":min(SKUNearestPod),
                                          "distance":SKUNearestPod[min(SKUNearestPod)]})
            #possiblePodsDistances[SKU] = minDist
        # for all SKUs, find the one whose nearest pod is nearest to the current position
        possiblePodsDistances = pd.DataFrame(possiblePodsDistances)
        nextSKU = possiblePodsDistances[possiblePodsDistances.distance == possiblePodsDistances.distance.min()]
        # add the distance to the respective pod to the total path distance
        distance[(str(route[-1]), str(int(nextSKU.pod)))] = round(float(nextSKU.distance), 1)
        # remove all items from the order that are in the pod and the order
        nextPod = podCopy[int(nextSKU.pod)]
        # for each SKU in the order, check whether items of it are in the next pod
        # first iterate over all the items in the order
        picked = []
        for index, row in order.iterrows():
            # next, check if each item is in the pod
            if index in nextPod.index:
                # if it is in the pod, determine whether all items from the order can be
                # picked from the pod, or whether there are more items in the pod than
                # there are in the order
                itemsToPick = min(row.quantity, nextPod.loc[index].Count)
                # remove the items that could be picked from the order
                if order.loc[index, "quantity"] == itemsToPick:
                    order = order.drop(index)
                else:
                    order.loc[index, "quantity"] -= itemsToPick
                # remove the items that could be picked from the order
                if nextPod.loc[index, "Count"] == itemsToPick:
                    nextPod = nextPod.drop(index)
                else:
                    nextPod.loc[index, "Count"] -= itemsToPick
                print("Removed " +  str(itemsToPick) + " item(s) of SKU " + str(index) + " from the order and from pod " + str(int(nextSKU.pod)) + ".")
                picked.append({"itemID":index, "count":itemsToPick})
        # add the pod to the route
        route.append(int(nextSKU.pod))
        routeInfo[int(nextSKU.pod)]= picked
    # add the last path of the route, from the last pod back to the station
    distance[(str(route[-1]), station)] = round(distance_ij[(str(route[-1]), station)], 1)
    route.append(station)
    return({"distances travelled":distance, "route taken":route, "route information":routeInfo, "altered podInfoDict":podCopy})