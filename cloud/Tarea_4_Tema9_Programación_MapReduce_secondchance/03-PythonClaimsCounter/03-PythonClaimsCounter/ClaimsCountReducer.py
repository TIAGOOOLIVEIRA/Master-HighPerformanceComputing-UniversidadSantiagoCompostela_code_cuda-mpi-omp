#!/usr/bin/env python2
import sys

def reducer():
    curr_count = 0
    curr_country = None
    curr_claims_total = 0
    

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        country, claims = line.split('\t')

        #to check if the claims is a number
        try:
            claims = int(claims)
        except ValueError:
            continue
        
        #to accumulate the claims for the same country
        if curr_country == country:
            curr_claims_total += claims
            curr_count += 1
        else:
            if curr_country:
                avg_claims = curr_claims_total / curr_count
                print("{0}\t{1:d}".format(curr_country, avg_claims))
            
            curr_country = country
            curr_claims_total = claims
            curr_count = 1
    
    #to write the average claims for the last country
    if curr_country:
        avg_claims = curr_claims_total / curr_count
        print("{0}\t{1:d}".format(curr_country, avg_claims))

if __name__ == "__main__":
    try:
        reducer()
    except Exception as e:
        #sys.stderr.write(f"Error - reducer: {str(e)}\n")
        sys.stderr.write("Error - reducer: {0}\n".format(str(e)))
        sys.exit(1)