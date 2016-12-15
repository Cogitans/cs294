After cloning/downloading this project, to run, cd into the code directory. Afterwards run

    python speaker_segment_net.py

In order to change the dataset that the generator is run on, in code/speaker_segment.py, change the TRAIN_GENERATOR/TEST_GENERATOR variables at the top. We have currently implemented two training sample generators and two test sample generators in data_gen.py

- Training
    * shakespeare_soft_train_gen
    * movie_soft_train_gen

- Test
    * shakespeare_soft_test_gen
    * movie_soft_test_gen