Association Rules - Medium Task
    Idea: Analyze web browsing histories to recommend relevant articles or websites to users.
    Input: Web logs containing user browsing sessions (user ID, website visited).
    Output: Association rules indicating browsing patterns and recommendations. make separate mapper.py reducer.py
ChatGPT
Below are the separate mapper.py and reducer.py scripts for implementing association rules on web browsing histories to recommend relevant articles or websites to users:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys

# Read input from STDIN
for line in sys.stdin:
    # Split the line into user ID and website visited
    user_id, website = line.strip().split('\t')

    # Emit user ID and website as key-value pairs
    print('%s\t%s' % (user_id, website))
reducer.py
python
Copy code
#!/usr/bin/env python
from itertools import combinations
from collections import defaultdict
import sys

# Initialize variables
current_user = None
websites = []

# Read input from STDIN
for line in sys.stdin:
    # Split the line into user ID and website visited
    user_id, website = line.strip().split('\t')

    # If the user changes, process the previous user's data
    if current_user and user_id != current_user:
        # Generate combinations of visited websites
        for subset in combinations(websites, 2):
            print('%s\t%s\t%d' % (' & '.join(sorted(subset)), 1))
        
        # Clear the list for the new user
        websites = []

    # Add the website visited to the list
    websites.append(website)
    current_user = user_id

# Process the last user's data
if current_user:
    for subset in combinations(websites, 2):
        print('%s\t%s\t%d' % (' & '.join(sorted(subset)), 1))
These scripts process the input web logs, emitting combinations of websites visited by users along with their frequencies. The mapper script simply reads input and emits the user ID and website visited. The reducer script aggregates the websites visited by each user and emits combinations of visited websites along with their counts. These scripts can be run in a Hadoop or MapReduce environment for analyzing large datasets of web browsing histories.





