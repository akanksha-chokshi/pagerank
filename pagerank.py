import os
import random
import re
import sys
from numpy.random import choice
import numpy as np


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print("PageRank Results from Sampling " + str(SAMPLES))
    for page in sorted(ranks):
        print("  " + page + ": " + str(ranks[page]))
    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print("  " + page + ": " + str(ranks[page]))


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_distribution = dict ()
    # if the page has atleast one link
    if corpus [page]:
        for element in corpus:
            prob_distribution [element]= (1 - damping_factor)/ len (corpus)
            # for every page linked by the current page
            if element in corpus [page]:
                prob_distribution [element] += damping_factor/ len (corpus[page])
    # if page has no outgoing links
    else:
        for element in corpus:
            prob_distribution [element] = 1/len(corpus)

    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initialising ranks to 0
    ranks = dict ()
    for element in corpus:
        ranks [element] = 0.0

    page = random.choice(list(corpus)) 
    # starting with a random page and sampling 'n' pages
    for i in range (n):
        if page is None: 
            page = random.choice(list(corpus.keys()))
        else:
            # increasing count for every page visited
            ranks[page] = ranks[page] + 1
            # generating a new sample of pages to sample
            current_rank = transition_model (corpus, page, damping_factor)
            choices = current_rank.keys()
            weights = current_rank.values()
            norm_weights = weights/ np.sum(weights)
            resample_counts = np.random.multinomial(1, norm_weights)

            chosen = []
            resample_index = 0
            for resample_count in resample_counts:
                for _ in range(resample_count):
                    chosen.append(choices[resample_index])
                resample_index += 1
            page = chosen [0]
        ranks [page] += 1
    # dividing count of times visited by number of pages
    for page in corpus:
        ranks[page] = ranks[page]/(2 * n)        
    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict ()
    newrank = dict ()
    threshold = 0.001
    n = len(corpus)

    # assigning each page an initial rank
    for page in corpus:
        pagerank[page] = 1/n

    repeat = True

    # for every page, calculating its PageRank based on the rank of all the pages that link to it
    while repeat:
        for page in pagerank:
            total = 0
            for link_page in corpus:
                if page in corpus[link_page]:
                    total += pagerank[link_page]/len(corpus[link_page])
                if not corpus[link_page]:
                    total += pagerank[link_page]/n
            newrank[page] = (1 - damping_factor)/n + (damping_factor * total)
        repeat = False
        # for every page, if the newly calculated value and previous rank vary by more than the threshold, repeat the process
        for page in pagerank:
            if abs(pagerank[page] - newrank [page]) > 0.001:
                repeat = True
            pagerank[page] = newrank[page]    
    return pagerank


if __name__ == "__main__":
    main()
