# Read in the bib database
#with open('mlabs.bbl') as bibtex_file:
#    bib_database = bibtexparser.load(bibtex_file)

# convenience variable to access papers
papers = bib_database.entries
N_bib  = len(bib_database.entries) # Total number of papers

# Split off individual properties as numpy arrays
IDs       = array([p['ID'] for p in papers])
abstracts = array([p['abstract'].replace('\n',' ') if p.has_key('abstract') else '' for p in papers])
eprints   = array([p['eprint'] if p.has_key('eprint') else '' for p in papers])
journals  = array([p['journal'] if p.has_key('journal') else '' for p in papers])
years     = array([int(p['year']) for p in papers]) # They all have years, thankfully

# List of keys to probe abstracts with
abstexts = ['neural network','support vector','nearest neighb','random forest','regression','PCA','cluster','deep learn','decision tree','gradient boost','generative adversarial','generalized linear','active learn','gaussian process','perceptron']

# Filter out blank abstracts
blanks = abstracts == ''
filt   = ~blanks

IDs    = IDs[filt]
abstracts = abstracts[filt]
eprints = eprints[filt]
journal = journals[filt]
years   = years[filt]

N = len(IDs) # Total number of papers under analysis

# Set up dictionary of masks for each key
select = {}

# Number of keys referenced by each paper
# When this quantity is zero, the paper does not reference any of the keys above
numrefs = np.zeros(N) 
# print sum(numrefs == 0) # This will print the number of such papers

for abstext in abstexts:
    select.update({abstext:[abstext.lower() in a.lower() for a in abstracts]})
    numrefs    = numrefs+array(select[abstext])*1.0

associated = numrefs > 0  # Those papers that are associated with at least one key
otherlist = ~associated   # Papers assigned to "other" category
select.update({'other':otherlist})
numrefs[otherlist] = 1

weight = {}
for abstext in abstexts+['other']:
    weight.update({abstext:array(select[abstext])*1.0/numrefs})

# Papers corresponding to individual keys can now be accessed with, e.g. :
# IDs[select['neural network']]

yearbins = arange(min(years),max(years)+2,1)-0.5
yeararr  = arange(min(years),max(years)+1)

yearhist = {}
yearsum  = zeros(len(yeararr))
for abstext in abstexts+['other']:
    yearhist.update({abstext:histogram(years[select[abstext]],bins=yearbins)[0]})
    yearsum = yearsum + yearhist[abstext]
