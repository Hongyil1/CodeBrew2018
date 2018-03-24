from pymongo import MongoClient
import datetime
URI = 'mongodb://HHATK:CBrew2018@ds123499.mlab.com:23499/cbrew2018'

connection = MongoClient(URI)
db = connection['cbrew2018']
print(db.authenticate('HHATK', 'CBrew2018'))

print(db)
collection = db['records']
print(collection)

#newUser = {"Date": "uniqueDate","Cases": 0,"Diagnosis": "DIAGNOSISTEXT","Symptoms": "SYMPTOMS TEXT"}
#print(collection.find_one_and_update({'name': 'uniqueName'},{'$inc': {'score': 1}}))
def findDate(date):
	return (collection.find_one({'Date': date}))
dates = "2003-02-23"
#print(collection.find_one({'Date': '2003-02-23'}))

#print(findDate(dates))
#date is date to update for, number is amount to increment by. can decrement by using -ve values
def incrementCaseNo(date, number):
	return (collection.find_one_and_update({'Date': date}, {'$inc': {'CaseNo': number}}))

def retrieveMostRecent(noRecords):
	return(collection.find(limit = noRecords).sort('Date', -1))#, sort = {'Date', -1}))#.limit(noRecords).sort('Date',-1))

def printMostRecent(noRecords):
	last10 = []
	for doc in retrieveMostRecent(noRecords):
		last10.append(doc)
		#print(doc)
	return last10
print (printMostRecent(10))
