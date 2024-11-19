#!/usr/bin/env python2
import sys
import csv

def mapper():
    reader = csv.reader(sys.stdin)

    for row in reader:
        #to ensure the file format on reading
        if len(row) < 9:
            continue
        
        #to skip the header row, move to next with data
        if "COUNTRY" in row and "CLAIMS" in row:
            continue
    
        #country\tclaims
        #removes any leading, and trailing whitespaces.
        country = row[4].strip()
        claims = row[8].strip()
        
        #to replace empty values with 0
        if not claims:
            claims = 0
        else:
            try:
                claims = int(claims)
            except ValueError:
                claims = 0
        
        if country:
            print("{}\t{}".format(country, claims))

if __name__ == "__main__":
    try:
        mapper()
    except Exception as e:
        sys.stderr.write("Error in mapper: {0}\n".format(str(e)))
        sys.exit(1)