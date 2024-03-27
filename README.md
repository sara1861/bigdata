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











   PageRank - Medium Task
    Idea: Determine the most influential nodes in a network of online forums based on user interactions and discussions.
    Input: Forum post data (user ID, post content, replies).
    Output: Ranking of forum threads or users based on their PageRank scores. separate mapper and reducer
ChatGPT
For implementing PageRank on a network of online forums based on user interactions and discussions, you'll typically want to represent the forum network as a graph where users and forum threads are nodes, and interactions (such as replies) are edges between them. Below are the separate mapper.py and reducer.py scripts for this task:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys

# Mapper function to emit graph structure
for line in sys.stdin:
    # Split the input into user ID, post ID, and replies
    user_id, post_id, replies = line.strip().split('\t')

    # Emit the post ID as the key with its associated user ID and replies
    print('%s\t%s\t%s' % (post_id, user_id, replies))

    # Emit the user ID as the key with its associated post ID and replies
    print('%s\t%s\t%s' % (user_id, post_id, replies))

# Initial PageRank values for posts and users
# Assuming equal initial PageRank for all nodes
print('%s\t%s\t%s' % ('#', 'post', 1.0))
print('%s\t%s\t%s' % ('#', 'user', 1.0))
reducer.py
python
Copy code
#!/usr/bin/env python
from collections import defaultdict
import sys

# Initialize variables
current_key = None
values = defaultdict(list)
d = 0.85  # Damping factor
N = 0     # Total number of nodes

# Function to calculate PageRank scores
def calculate_pagerank(scores, d, N):
    # Initial PageRank values
    pageranks = defaultdict(float)
    for node in scores:
        pageranks[node] = 1.0 / N

    # Iteratively calculate PageRank scores
    for _ in range(10):  # 10 iterations for convergence
        new_pageranks = defaultdict(float)
        for node in scores:
            new_pagerank = (1 - d) / N
            for neighbor in scores[node]:
                new_pagerank += d * pageranks[neighbor] / len(scores[neighbor])
            new_pageranks[node] = new_pagerank
        pageranks = new_pageranks

    return pageranks

# Read input from STDIN
for line in sys.stdin:
    # Split the line into key, node type, and value
    key, node_type, value = line.strip().split('\t')

    # If the key changes, process the previous key's data
    if current_key and key != current_key:
        # Process the accumulated values for the current key
        if current_key != '#':
            # Calculate PageRank scores
            pageranks = calculate_pagerank(values, d, N)

            # Emit PageRank scores
            for node, pagerank in pageranks.items():
                print('%s\t%s\t%s' % (current_key, node, pagerank))
        
        # Reset variables for the new key
        values = defaultdict(list)

    # Update current key
    current_key = key

    # Update values dictionary
    if node_type == 'post' or node_type == 'user':
        N += 1  # Increment total number of nodes
    values[node_type].append(value)

# Process the last key's data
if current_key != '#':
    # Calculate PageRank scores
    pageranks = calculate_pagerank(values, d, N)

    # Emit PageRank scores
    for node, pagerank in pageranks.items():
        print('%s\t%s\t%s' % (current_key, node, pagerank))
These scripts implement the Mapper and Reducer functions for PageRank computation on a network of online forums based on user interactions and discussions. The mapper script emits the graph structure and initializes PageRank values for posts and users. The reducer script accumulates the graph data and calculates PageRank scores iteratively for each node type (posts and users). Finally, it emits the PageRank scores for each node. These scripts can be run in a Hadoop or MapReduce environment for analyzing large datasets of forum post interactions.




