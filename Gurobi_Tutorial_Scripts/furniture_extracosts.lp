\ Model Furniture
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  45 make[chair] + 80 make[table] - 20 extra[mahogany] - 30 extra[labor]
Subject To
 R[mahogany]: 5 make[chair] + 20 make[table] - extra[mahogany] <= 400
 R[labor]: 10 make[chair] + 15 make[table] - extra[labor] <= 450
Bounds
End
