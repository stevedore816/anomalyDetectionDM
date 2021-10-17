import pandas as pd

"Imports csv files for all of the different attack types"
struct = pd.read_csv('Data/demonstrate_structure.csv')
benign_traffic = pd.read_csv('Data/benign_traffic.csv')

attack_vector_traffic = pd.concat(map(pd.read_csv,['Data/gafgyt_attacks/combo.csv','Data/gafgyt_attacks/junk.csv','Data/gafgyt_attacks/scan.csv','Data/gafgyt_attacks/tcp.csv','Data/gafgyt_attacks/udp.csv']), ignore_index= True)

"""To later combine the two files I created two rows set to 0 or 1 based on if there attacks or not"""
benignRow = []
for i in range (0, len(benign_traffic)):
    benignRow.append(0)

attackVectorRow = []

for i in range (0, len(attack_vector_traffic)):
    attackVectorRow.append(1)

"""I then add these rows and combine them together to get the final file that has them listed as 0 or 1 which can be learned with anomaly detection."""
benign_traffic['Attack'] = benignRow

attack_vector_traffic['Attack'] = attackVectorRow

final_file = pd.concat([benign_traffic, attack_vector_traffic],ignore_index=True)
print(final_file)
final_file.to_csv(r'Data/preprocessed.csv',index = True)