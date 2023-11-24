\ Model Get_XandY
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  Mx[0]
Subject To
 b: X[0] <= 30
 c: X[0] <= 20
 0: - X[0] + Mx[0] <= 0
 0: Mx[0] <= 20
 0: - X[0] + Mx[0] + 10000 B1[0] >= 0
 0: Mx[0] + 10000 B2[0] >= 20
 0: B1[0] + B2[0] = 1
 g: Z[0] + Z[1] + Z[2] + Z[3] + Z[4] + Z[5] + Z[6] + Z[7] + Z[8] + Z[9]
   + Z[10] + Z[11] + Z[12] + Z[13] + Z[14] + Z[15] + Z[16] + Z[17] + Z[18]
   + Z[19] + Z[20] + Z[21] + Z[22] + Z[23] + Z[24] + Z[25] + Z[26] + Z[27]
   + Z[28] + Z[29] + Z[30] + Z[31] + Z[32] + Z[33] + Z[34] <= 0
Bounds
Binaries
 B1[0] B2[0] Z[0] Z[1] Z[2] Z[3] Z[4] Z[5] Z[6] Z[7] Z[8] Z[9] Z[10] Z[11]
 Z[12] Z[13] Z[14] Z[15] Z[16] Z[17] Z[18] Z[19] Z[20] Z[21] Z[22] Z[23]
 Z[24] Z[25] Z[26] Z[27] Z[28] Z[29] Z[30] Z[31] Z[32] Z[33] Z[34]
Generals
 X[0] Mx[0]
End
