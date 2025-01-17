# Continous-Sign-Language-Translator-
Sign language interpreters are currently required for interpreting the speech impaired people. This skill-based job of interpreters is cumbersome and hence the number of interpreters per capita across majority countries are very low or decreasing. We aim to harness technology in developing a powerful continuous sign language gestures recognition system. This computer vision-based approach will be used to recognise Argentinian sign language gestures from a video. Translating these sign language gestures is considered a monumental task in this field.  The project proposes to investigate whether sign language gestures can be recognised by using a trained modified Inception V3 working as a feature selector and classifier, with a LTSM Recurrent Neural Network. Two separate approaches have been applied to recognise the Argentinian gestures. The Global Max Pooling approach outperforms the SoftMax approach, with a model accuracy of 86.10% on validation set and 75.2% on test set. Using the Inception V3 model as a feature extractor for LTSM RNN worked more efficiently and produced better results than using the Inception V3 model as a classifier. These results show the effectiveness of the research conducted. This research will help in classifying and recognising continuous sign language gestures based on machine vision. This in turn will assist people that are affected by speech and hearing impairment in understanding, translating and recognising sign gestures. 



Requirements :
--------------
  Download LSA64 (all_cut version) ~ 1.5GB from http://facundoq.github.io/datasets/lsa64/
  or use the download link : https://mega.nz/#!FQJGCYba!uJKGKLW1VlpCpLCrGVu89wyQnm9b4sKquCOEAjW5zMo
  
  
Updates :
---------
    - 1_process-lsa64.py : Copy videos to new directories (classes)
            
    - 2_process_LS64_vid2frames.py  : Converts videos to Frames
                                      For each single video --> frames in a sud-directory or the class directories
