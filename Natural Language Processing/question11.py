# How many 3-grams can be generated from this sentence "I love New York style pizza"?

# We will use CountVectorizer package to demonstrate how to use N-Gram with Scikit-Learn.
# CountVectorizer converts a collection of text documents to a matrix of token counts.
# In our case, there is only one document.

from sklearn.feature_extraction.text import CountVectorizer

# N-gram_range specifies the lower and upper boundary on the range of N-gram tokens to be extracted.
# For our example, the range is from 3 to 3.
# We have to specify the token_pattern because, by default, CountVectorizer treats single character
# words as stop words.
vectorizer = CountVectorizer(ngram_range=(3, 3),
                             token_pattern = r"(?u)\b\w+\b",
                             lowercase=False)

# Now, let's fit the model with our input text
vectorizer.fit(["I love New York style pizza"])

# This will populate vectorizer's vocabulary_ dictionary with the tokens.
# Let's see the results of this vocabulary
print(vectorizer.vocabulary_.keys())
