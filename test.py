import os
for filename in os.listdir('train'):
 os.system("cat temp | cat >> train/"+filename+"")

