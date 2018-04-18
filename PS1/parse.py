import csv

data_file = 'house_votes_84.data'

def parse(filename):
  '''
  takes a filename and returns attribute information and all the data in array of dictionaries
  '''
  # initialize variables

  out = []  
  csvfile = open(filename,'rb')
  fileToRead = csv.reader(csvfile)

  headers = fileToRead.next()

  # iterate through rows of actual data
  for row in fileToRead:
    out.append(dict(zip(headers, row)))

  return out

examples = parse(data_file)
#print(type(data))
#print(len(data[0]))
#print(data[0])
#print(data[1])
#print(data[0].keys()[0])
#print(data[0]["handicapped-infants"])