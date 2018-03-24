import csv
import matplotlib.pyplot as plt

inptA= 'data.csv' 
csvopen= open(inptA,"r") 
file= csv.reader(csvopen)
fileT= list(zip(*file)) #transpose to make it easier to iterate over
case=[float(x) for x in fileT[1][1:]]
last10=[case[-10:]]
#MAX 10 baselines!
baselines=[]

for i in range(len(case)):
	#if 10 baselines already recorded, overwrite the oldest baseline.
	if i > 9:
		baseline = sum(case[i-9:i+1])/10
		remainder= i %10
		baselines[remainder] = baseline
	#if baselines list not yet full, append baseline to end of list.
	else:
		baseline = sum(case[:i+1])/(i+1)
		baselines.append(baseline)
	
	#Situation has improved
	if (case[i]*1.05) < baseline:
		plt.plot(case[i],baseline, marker='o', c = 'green')
	#Situation has Worsened!
	elif (case[i]) > baseline*1.05:
		plt.plot(case[i],baseline, marker='o', c = 'red')
	#Situation hasn't changed
	else:
		plt.plot(case[i],baseline, marker='o', c = 'yellow')
		
ax.set_xlabel('Disease Rate')
ax.set_ylabel('Baseline')
ax.

plt.axis([0,10.1,0,1])
plt.show(block=True)