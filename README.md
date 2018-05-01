# restaurant-reviews

The code will run several regression and classification models based on features selected with the options below.

### Command Line Options: 

* '--name' - name of the folder that will be created where images and results will appear (default is 'test')
* '--nRevs' - number of reviews to use
* '--nUnigrams'  - maximum number of unigram features (only needed if minDF is not used)
* '--nBigrams' - same as nUnigrams but for bigrams
* '--split' - boolean switch that will split the unigrams and bigrams into their own vectorizers so you get an even number of features (default is false)
* '--minDF' - instead of taking a max number of features, an ngram must have at least the number of occurrences that you set here
* '--onlyUnigrams' - boolean switch to include unigrams (must include --split to use this)
* '--onlyBigrams' - boolean switch to include bigrams (must include --split to use this)
* '--allCaps' - boolean switch that will include all-capital words with at least 75 occurrences
* '--stats' - boolean switch that will include a feature that contains the length of the review
* '--binary' - boolean switch that will make the classification binary instead of multiclass


The rest are the weights feature sets. They're optional, but if you use them set between 0 and 1.0 (default is 1.0):

* '--unigramW' - weight for unigrams
* '--bigramW' - weight for bigrams
* '--unigramAndBigramW' - weight for unigrams and bigrams if you are not splitting them
* '--allCapsW' - weight for all caps
* '--stats' - weight for stats
