import spacy

nlp = spacy.load('en_core_web_sm')


gardenpathSentences = ["The girl told the story cried",
                       "Helen is expecting tomorrow to be a bad day",
                       "Mary gave the child a Band-Aid.",
                       "That Jill is never here hurts.",
                       " The cotton clothing is made of grows in Mississippi."
                       ]


doc = nlp(" ".join(gardenpathSentences))

# Tokenisation using spacy library
[token.orth_ for token in doc]
print([(token, token.orth_, token.orth) for token in doc])


# Named entity recognition
nlp_garden = nlp(''.join(gardenpathSentences))
print([(i, i.label_, i.label) for i in nlp_garden.ents])

# entity recognition
for sentence in gardenpathSentences:
    doc = nlp(sentence)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    print(entities)

# Get an explanation of an entity and print it
entity_gpe = spacy.explain("GPE")
print(f"GPE:{entity_gpe}")


# "The girl told the story cried",
# In this garden sentence there is no entity, the sentence does not refer to any specific named entity.
# Explanation - the girl who told the story was crying. In other words, "cried" is a modifier that describes the 
# state of the girl while she was telling the story.
# There is no entity in the sentence as such i cannot answer the second question

# "Helen is expecting tomorrow to be a bad day",
# Helen and tomorrow are identified entities
# Explanation - Helen is anticipating the following day will be unpleasant.
# There are no words associated with the entities however the sentence made sense upon the first reading


