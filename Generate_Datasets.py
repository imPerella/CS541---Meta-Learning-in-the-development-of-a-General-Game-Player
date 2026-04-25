from Generation_Functions import *

def dataset(variant_values:list = [10,10,10,10,10,10]):
   var1, var2, var3, var4, var5, var6 = variant_values
   X1,Y1 = generate_tic_tac_toe(num_variants=var1)
   X2,Y2 = generate_connect_four(num_variants=var2)
   X3,Y3 = generate_othello(num_variants=var3)
   X4,Y4 = generate_ataxx(num_variants=var4)
   X5,Y5 = generate_checkers(num_variants=var5, keep_pieces=True)
   X6,Y6 = generate_checkers(num_variants=var6, keep_pieces=False)

   X = X1 + X2 + X3 + X4 + X5 + X6
   Y = Y1 + Y2 + Y3 + Y4 + Y5 + Y6

   return X,Y