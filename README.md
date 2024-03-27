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


Frequent Item Sets - Easy Task
    Idea: Identify frequently co-purchased items in a retail store to optimize product placement on shelves.
    Input: Point-of-sale transaction data (transaction ID, purchased items).
    Output: List of frequently co-purchased item sets and their support counts.

ChatGPT
For identifying frequently co-purchased items in a retail store using the Apriori algorithm, you'll need to implement a Mapper and Reducer for the task. Below are the separate mapper.py and reducer.py scripts for this task:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys
from itertools import combinations

# Read input from STDIN
for line in sys.stdin:
    # Split the line into transaction ID and purchased items
    transaction_id, items = line.strip().split('\t')
    items = items.split(',')

    # Generate item pairs from the purchased items
    for pair in combinations(items, 2):
        # Sort the pair to avoid duplicates
        pair = sorted(pair)
        # Emit the pair as key with count 1
        print('%s\t%s' % (','.join(pair), 1))
reducer.py
python
Copy code
#!/usr/bin/env python
from collections import defaultdict
import sys

# Initialize variables
current_pair = None
current_count = 0
pair_count = defaultdict(int)

# Read input from STDIN
for line in sys.stdin:
    # Split the line into key and count
    pair, count = line.strip().split('\t')

    # Convert count to integer
    count = int(count)

    # If the pair changes, process the previous pair's data
    if current_pair and pair != current_pair:
        # Emit the pair and its count
        print('%s\t%s' % (current_pair, current_count))
        # Reset count for the new pair
        current_count = 0
    
    # Update current pair and count
    current_pair = pair
    current_count += count

# Process the last pair's data
if current_pair:
    print('%s\t%s' % (current_pair, current_count))


Google Matrix - Easy Task
    Idea: Analyze communication patterns in a corporate email network to identify key employees for organizational restructuring.
    Input: Email communication data (sender, receiver, timestamp).
    Output: Ranking of employees based on their importance computed using the Google Matrix.
ChatGPT
To compute the Google Matrix for analyzing communication patterns in a corporate email network and identifying key employees, you can use the Mapper and Reducer approach. Below are the separate mapper.py and reducer.py scripts for this task:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys

# Read input from STDIN
for line in sys.stdin:
    # Split the line into sender, receiver, and timestamp
    sender, receiver, timestamp = line.strip().split('\t')

    # Emit sender-receiver pair as key with count 1
    print('%s\t%s' % (sender, receiver))
reducer.py
python
Copy code
#!/usr/bin/env python
from collections import defaultdict
import sys

# Initialize variables
adjacency_list = defaultdict(list)
pageranks = defaultdict(float)
damping_factor = 0.85
N = 0  # Total number of employees

# Read input from STDIN
for line in sys.stdin:
    # Split the line into sender and receiver
    sender, receiver = line.strip().split('\t')

    # Build adjacency list
    adjacency_list[sender].append(receiver)

    # Increment the total number of employees
    N += 1

# Initialize PageRank values
for employee in adjacency_list:
    pageranks[employee] = 1.0 / N

# Perform PageRank iterations
for _ in range(10):  # 10 iterations for convergence
    new_pageranks = defaultdict(float)
    for sender in adjacency_list:
        new_pagerank = (1 - damping_factor) / N
        for receiver in adjacency_list[sender]:
            new_pagerank += damping_factor * pageranks[receiver] / len(adjacency_list[receiver])
        new_pageranks[sender] = new_pagerank
    pageranks = new_pageranks

# Emit the ranking of employees based on their importance
for employee, importance in sorted(pageranks.items(), key=lambda x: x[1], reverse=True):
    print('%s\t%s' % (employee, importance))
These scripts implement the Mapper and Reducer for computing the Google Matrix on email communication data to identify key employees in a corporate network. The mapper script emits sender-receiver pairs from the email communication data. The reducer script builds the adjacency list from the emitted pairs, computes PageRank scores iteratively, and finally emits the ranking of employees based on their importance computed using the Google Matrix. These scripts can be executed in a Hadoop or MapReduce environment to analyze large datasets of email communication data.




write one for word count too
ChatGPT
Certainly! Below are the separate mapper.py and reducer.py scripts for implementing word count:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys

# Read input from STDIN
for line in sys.stdin:
    # Split the line into words
    words = line.strip().split()

    # Emit each word as key with count 1
    for word in words:
        print('%s\t%s' % (word, 1))
reducer.py
python
Copy code
#!/usr/bin/env python
from collections import defaultdict
import sys

# Initialize variables
word_counts = defaultdict(int)

# Read input from STDIN
for line in sys.stdin:
    # Split the line into word and count
    word, count = line.strip().split('\t')

    # Update word counts
    word_counts[word] += int(count)

# Emit the word counts
for word, count in word_counts.items():
    print('%s\t%s' % (word, count))
These scripts implement the Mapper and Reducer for counting the occurrences of each word in a text corpus. The mapper script reads input text and emits each word encountered along with a count of 1. The reducer script aggregates the counts for each word and emits the total count for each word. When executed in a Hadoop or MapReduce environment, these scripts can efficiently perform word count on large text datasets.

User
one for LSH
ChatGPT
Locality Sensitive Hashing (LSH) is a technique used for approximate nearest neighbor search in high-dimensional spaces. Below are the separate mapper.py and reducer.py scripts for implementing LSH:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys

# Constants for LSH
BANDS = 5
ROWS_PER_BAND = 4

# Function to hash a vector using LSH
def lsh_hash(vector):
    hash_value = ''
    for i in range(BANDS):
        band_hash = hash(tuple(vector[i * ROWS_PER_BAND:(i + 1) * ROWS_PER_BAND])) % 1000
        hash_value += str(band_hash) + ','
    return hash_value[:-1]

# Read input from STDIN
for line in sys.stdin:
    # Split the line into data point ID and vector
    data_point_id, vector_str = line.strip().split('\t')

    # Convert vector string to list of integers
    vector = [int(x) for x in vector_str.split(',')]

    # Apply LSH to the vector and emit hash value along with data point ID
    hash_value = lsh_hash(vector)
    print('%s\t%s' % (hash_value, data_point_id))
reducer.py
python
Copy code
#!/usr/bin/env python
import sys

# Read input from STDIN
for line in sys.stdin:
    # Split the line into hash value and data point ID
    hash_value, data_point_id = line.strip().split('\t')

    # Emit hash value and associated data point ID
    print('%s\t%s' % (hash_value, data_point_id))
These scripts implement the Mapper and Reducer for Locality Sensitive Hashing (LSH). The mapper script hashes the input vectors using LSH and emits the hash value along with the corresponding data point ID. The reducer script simply forwards the hash value and associated data point ID to the output. In a full LSH implementation, these outputs would be processed further to identify approximate nearest neighbors. These scripts can be run in a Hadoop or MapReduce environment for analyzing large datasets requiring approximate nearest neighbor search.





ONE FOR CONFIDENCE INTERVAL AND INTEREST

ChatGPT
Confidence intervals are statistical tools used to estimate the range within which a population parameter lies with a certain level of confidence. Below are the separate mapper.py and reducer.py scripts for calculating confidence intervals and interest.

mapper.py (for Confidence Interval)
python
Copy code
#!/usr/bin/env python
import sys
import math

# Read input from STDIN
for line in sys.stdin:
    # Split the line into value and count
    value, count = line.strip().split('\t')

    # Convert count to integer
    count = int(count)

    # Emit the value and count
    print('%s\t%s' % (value, count))

    # Also emit (value^2) and count
    print('%s\t%s' % (str(float(value) ** 2), count))
reducer.py (for Confidence Interval)
python
Copy code
#!/usr/bin/env python
import sys
import math

# Initialize variables
sum_values = 0
sum_squared_values = 0
total_count = 0

# Read input from STDIN
for line in sys.stdin:
    # Split the line into value and count
    value, count = line.strip().split('\t')

    # Convert count to integer
    count = int(count)

    # Update the sums and total count
    sum_values += float(value) * count
    sum_squared_values += float(value) ** 2 * count
    total_count += count

# Calculate mean and standard deviation
mean = sum_values / total_count
variance = (sum_squared_values / total_count) - (mean ** 2)
std_deviation = math.sqrt(variance)

# Calculate confidence interval
confidence_interval = 1.96 * (std_deviation / math.sqrt(total_count))

# Emit the mean and confidence interval
print('Mean\t%s' % mean)
print('Confidence Interval\t%s' % confidence_interval)
mapper.py (for Interest)
python
Copy code
#!/usr/bin/env python
import sys

# Read input from STDIN
for line in sys.stdin:
    # Split the line into key and value
    key, value = line.strip().split('\t')

    # Emit the key and value
    print('%s\t%s' % (key, value))
reducer.py (for Interest)
python
Copy code
#!/usr/bin/env python
import sys

# Initialize variables
sum_values = 0
total_count = 0

# Read input from STDIN
for line in sys.stdin:
    # Split the line into key and value
    key, value = line.strip().split('\t')

    # Convert value to float
    value = float(value)

    # Update the sum and total count
    sum_values += value
    total_count += 1

# Calculate interest
interest = sum_values / total_count

# Emit the interest
print('Interest\t%s' % interest)
These scripts implement the Mapper and Reducer for calculating confidence intervals and interest. The confidence interval scripts calculate the mean and confidence interval of a dataset, while the interest scripts calculate the interest based on the average value of a dataset. These scripts can be run in a Hadoop or MapReduce environment for analyzing large datasets and calculating statistical properties.

ONE FOR BASIC ASSOCIATION RULE
ChatGPT
To implement a basic Association Rule mining algorithm in a MapReduce setting, we need two main components: mapper.py and reducer.py. Here's how you can create these scripts:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys
from itertools import combinations

# Read input from STDIN
for line in sys.stdin:
    # Split the line into transaction and items
    transaction, items = line.strip().split('\t')

    # Split items into a list
    items_list = items.split(',')

    # Generate item pairs from the transaction
    for pair in combinations(items_list, 2):
        # Sort the pair to avoid duplicates
        pair = sorted(pair)
        # Emit the pair as key with count 1
        print('%s\t%s' % (','.join(pair), 1))
reducer.py
python
Copy code
#!/usr/bin/env python
from collections import defaultdict
import sys

# Initialize variables
current_key = None
item_count = defaultdict(int)
total_transactions = 0

# Read input from STDIN
for line in sys.stdin:
    # Split the line into key and count
    key, count = line.strip().split('\t')

    # If the key changes, process the previous key's data
    if current_key and key != current_key:
        # Calculate support for the current itemset
        support = item_count[current_key] / total_transactions

        # If support is greater than a threshold, emit association rule
        if support >= 0.1:  # Example threshold (can be adjusted)
            itemset = current_key.split(',')
            antecedent = itemset[0]
            consequent = itemset[1]
            confidence = item_count[current_key] / item_count[antecedent]
            print('%s -> %s\t%s\t%s' % (antecedent, consequent, support, confidence))
        
        # Reset count for the new key
        item_count = defaultdict(int)

    # Update current key
    current_key = key

    # Update item count and total transactions
    item_count[key] += int(count)
    total_transactions += 1

# Process the last key's data
if current_key:
    # Calculate support for the last itemset
    support = item_count[current_key] / total_transactions
    
    # If support is greater than a threshold, emit association rule
    if support >= 0.1:  # Example threshold (can be adjusted)
        itemset = current_key.split(',')
        antecedent = itemset[0]
        consequent = itemset[1]
        confidence = item_count[current_key] / item_count[antecedent]
        print('%s -> %s\t%s\t%s' % (antecedent, consequent, support, confidence))
These scripts implement the Mapper and Reducer for basic Association Rule mining in a MapReduce framework. The mapper script generates item pairs from each transaction and emits them with a count of 1. The reducer script aggregates the counts for each itemset, calculates support and confidence, and emits association rules for those above a certain threshold. Adjust the threshold as needed for your use case. These scripts can be executed in a Hadoop or MapReduce environment to analyze large datasets for frequent itemsets and generate association rules.

DO ONE FOR JACCARD SIMILARITY
ChatGPT
To compute Jaccard similarity between sets of items in a MapReduce setting, we can create a mapper that emits all pairs of sets along with their intersection size, and then a reducer that calculates the Jaccard similarity for each pair. Here's how you can implement this:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys
from itertools import combinations

# Read input from STDIN
for line in sys.stdin:
    # Split the line into transaction ID and items
    transaction_id, items_str = line.strip().split('\t')

    # Convert the items string into a set
    items = set(items_str.split(','))

    # Generate pairs of sets from the input transaction
    for pair in combinations(items, 2):
        # Sort the pair to avoid duplicates
        pair = sorted(pair)

        # Emit the pair of items as key and their intersection size as value
        print('%s\t%s' % (','.join(pair), len(items)))
reducer.py
python
Copy code
#!/usr/bin/env python
import sys

# Initialize variables
current_pair = None
current_intersection_size = 0
current_union_size = 0

# Read input from STDIN
for line in sys.stdin:
    # Split the line into key and value
    pair, size = line.strip().split('\t')

    # Convert size to integer
    size = int(size)

    # If the pair changes, calculate Jaccard similarity
    if current_pair and pair != current_pair:
        jaccard_similarity = current_intersection_size / current_union_size

        # Emit the pair and its Jaccard similarity
        print('%s\t%s' % (current_pair, jaccard_similarity))

        # Reset variables for the new pair
        current_intersection_size = 0
        current_union_size = 0

    # Update current pair
    current_pair = pair

    # Update intersection size
    current_intersection_size += size

    # Update union size
    current_union_size += size

# Calculate Jaccard similarity for the last pair
if current_pair:
    jaccard_similarity = current_intersection_size / current_union_size
    print('%s\t%s' % (current_pair, jaccard_similarity))
These scripts implement the Mapper and Reducer for computing Jaccard similarity between sets of items in a MapReduce framework. The mapper generates pairs of items from each input transaction and emits them with the size of the intersection of the corresponding sets. The reducer calculates the Jaccard similarity for each pair based on the intersection and union sizes and emits the result. These scripts can be run in a Hadoop or MapReduce environment to analyze large datasets for similarity between sets. Adjust the input format and delimiter (\t in this case) as per your data


ONE FOR PAGE RANK

ChatGPT
Implementing PageRank in a MapReduce setting involves multiple iterations of calculating the PageRank scores for each node based on the contributions from incoming edges. Below are the mapper.py and reducer.py scripts for the PageRank algorithm:

mapper.py
python
Copy code
#!/usr/bin/env python
import sys

# Read input from STDIN
for line in sys.stdin:
    # Split the line into node ID and its adjacency list
    node_id, adjacency_list = line.strip().split('\t')

    # Parse the adjacency list
    adjacency_list = eval(adjacency_list)

    # Emit the node ID with its adjacency list
    print('%s\t%s' % (node_id, adjacency_list))

    # Emit each neighbor with its contribution to the node's PageRank
    if adjacency_list:
        contribution = 1 / len(adjacency_list)
        for neighbor in adjacency_list:
            print('%s\t%s' % (neighbor, contribution))
reducer.py
python
Copy code
#!/usr/bin/env python
import sys

# Initialize variables
current_node = None
pagerank = 0.0
adjacency_list = []

# Read input from STDIN
for line in sys.stdin:
    # Split the line into node ID and value
    node_id, value = line.strip().split('\t')

    # If the node ID changes, emit the node ID with its updated PageRank
    if current_node and node_id != current_node:
        print('%s\t%s' % (current_node, pagerank))

        # If the node has outgoing edges, emit its adjacency list
        if adjacency_list:
            print('%s\t%s' % (current_node, adjacency_list))

        # Reset variables for the new node
        pagerank = 0.0
        adjacency_list = []

    # Update current node ID
    current_node = node_id

    # If the value is a float, it represents a contribution to PageRank
    if '.' in value:
        pagerank += float(value)
    # If the value is a list, it represents an adjacency list
    else:
        adjacency_list = value

# Emit the last node's PageRank and adjacency list (if any)
if current_node:
    print('%s\t%s' % (current_node, pagerank))
    if adjacency_list:
        print('%s\t%s' % (current_node, adjacency_list))
These scripts implement the Mapper and Reducer for the PageRank algorithm in a MapReduce framework. The mapper emits each node with its adjacency list and the contribution from each neighbor. The reducer aggregates the contributions to calculate the new PageRank for each node. This process is iterated multiple times until convergence to obtain the final PageRank scores for each node. These scripts can be executed in a Hadoop or MapReduce environment to analyze large graphs and calculate PageRank scores.








