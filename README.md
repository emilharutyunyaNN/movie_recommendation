Recommendation system using NCF (Neural Collaborative Filtering) and MF (Matrix Factorization)

        This project's focus is making recommendations based on already available movie rating dataset with ratings with some missing. We are working with MovieLens 100K dataset. We do data preprocessing and split the dataset into training and testing sets. Then we train our model on the training set, and make predictions, i.e. recommendations for the test set. 

How to run

        I had some problems installing Lightfm with pip, so I created a tensorflow eenvironment in anaconda prompt 
        then:
            - conda activate tensorflow
            - cd /path/to/the/directory
            - python3 pipeline.py

        After running we see the training process and the resulting recommendations on the test set.

        Or if there are no problems with the installation of lightfm
            - python3 pipeline.py

Contributions

        My code is a simple implementation of a recommendation system model, it generates somewhat accurate predictions, but it can be further improved, to make it a fully functional recommendation systems that takes into consideration other parameters. It can then be used by webpages to recommend movies based on ratings and even number of views.

    