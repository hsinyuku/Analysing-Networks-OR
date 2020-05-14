\ Model Furniture
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  45 make[chair] + 80 make[table]
Subject To
 R[mahogany]: 5 make[chair] + 20 make[table] >= 400
 R[labor]: 10 make[chair] + 15 make[table] >= 450
Bounds
 make[chair] <= 200
 make[table] <= 150
End
