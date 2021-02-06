import project as pj
from sklearn import metrics

val = input("Enter your id: ")
obama_filtered = pj.filter_obama(r"/Users/ashwin/Desktop/training-Obama-Romney-tweets.xlsx", dropnan=True)
romney_filtered = pj.filter_romney(r"/Users/ashwin/Desktop/training-Obama-Romney-tweets.xlsx", dropnan=True)
obama_filtered = pj.PreProcessing(obama_filtered, pj.negative_word_conversion())
romney_filtered = pj.PreProcessing(romney_filtered, pj.negative_word_conversion())

obama_test_filtered = pj.obama_testing(r"/Users/ashwin/Desktop/final-testData-no-label-Obama-tweets.xlsx")
romney_test_filtered = pj.romney_testing(r"/Users/ashwin/Desktop/final-testData-no-label-Romney-tweets.xlsx")
obama_test_filtered = pj.PreProcessing(obama_test_filtered, pj.negative_word_conversion())
romney_test_filtered = pj.PreProcessing(romney_test_filtered, pj.negative_word_conversion())

train_model_obama = pj.model("svm")
train_model_obama.fit(obama_filtered['Annotated Tweet'], obama_filtered['Class'])
predicted_obama = train_model_obama.predict(obama_test_filtered['Annotated Tweet'])

train_model_romney = pj.model("svm")
train_model_romney.fit(romney_filtered['Annotated Tweet'], romney_filtered['Class'])
predicted_romney = train_model_romney.predict(romney_test_filtered['Annotated Tweet'])

pj.print_class(obama_test_filtered, predicted_obama, "/Users/ashwin/Desktop/obama.txt", val)
pj.print_class(romney_test_filtered, predicted_romney, "/Users/ashwin/Desktop/romney.txt", val)
