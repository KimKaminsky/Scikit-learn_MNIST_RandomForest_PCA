# Scikit-learn_MNIST_RandomForest_PCA
Assignment for Machine Learning course comparing the performance of Random Forest on the MNIST dataset with PCA applied and not applied.

The purpose of this study was to determine the advantages of using Principal Component Analysis (PCA) with regards to both processing time and model accuracy. Two models were compared: Random Forest (RF) run on a full dataset and RF run on a dimensionally reduced version of the dataset produced using PCA. The dataset considered was the MNIST, which consists of a set of 70,000 handwritten digits by high school students and employees of the US Census Bureau. This dataset is commonly used to test new machine learning algorithms. The models were compared using the F1-score, which is the harmonic mean of precision and recall.

Note: This was an assignment from a machine learning course. Random Forest tends not to suffer from the curse of dimensionality since each tree uses a subset of features which reduces the dimensionality. If this assignment had been comparing a single decision tree or some other machine learning methods such as linear regression, k-nearest neighbors, or k-means then the PCA probably would have had a beneficial effect. In this case, it did not.