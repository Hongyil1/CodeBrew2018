import csv 

inptA= 'cdc_zika.csv' 
csvopen= open(inptA,"r", ) 
file= csv.reader(csvopen)

#Params
value=["cases"]
date=["date"]
location=["location"]
fieldCode=["data_field"]

fileT= list(zip(*file)) #transpose to make it easier to iterate over
for i in range(len(fileT)):
	if "location" in fileT[i]:
		for j in range(len(fileT[i])):
			if "Argentina-Buenos_Aires" in fileT[i][j]:
				location.append(fileT[i][j])
				value.append(fileT[7][j])
				date.append(fileT[0][j])
				fieldCode.append(fileT[3][j])


#create list to order new CSV
listy=[location,date,value,fieldCode]
listy=zip(*listy)

#output to new CSV
output= open("ZikaClean.csv", 'w')

for row in listy:
	for column in row:
		output.write('{},'.format(column))
	output.write('\n')
output.close()

#